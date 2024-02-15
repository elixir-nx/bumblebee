defmodule Bumblebee.Text.TextGeneration do
  @moduledoc false

  alias Bumblebee.Shared
  alias Bumblebee.Text

  def generation(model_info, tokenizer, %Text.GenerationConfig{} = generation_config, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :compile,
        defn_options: [],
        preallocate_params: false,
        stream: false,
        stream_done: false
      ])

    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [
      :for_conditional_generation,
      :for_causal_language_modeling
    ])

    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size, :sequence_length])
        |> Shared.require_options!([:batch_size, :sequence_length])
      end

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    tokenizer =
      Bumblebee.configure(tokenizer,
        length: sequence_length,
        pad_direction: :left,
        return_token_type_ids: false,
        return_length: true
      )

    generate_fun =
      Bumblebee.Text.Generation.build_generate(model, spec, generation_config,
        ignore_output: opts[:stream]
      )

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32),
              "seed" => Nx.template({batch_size}, :s64)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          generate_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      if opts[:stream] do
        Shared.validate_input_for_stream!(input)
      end

      {inputs, multi?} = Shared.validate_serving_input!(input, &validate_input/1)

      texts = Enum.map(inputs, & &1.text)
      seed = Enum.map(inputs, & &1.seed) |> Nx.tensor(backend: Nx.BinaryBackend)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts)
        end)

      {input_length, inputs} = Map.pop!(inputs, "length")
      input_padded_length = Nx.axis_size(inputs["input_ids"], 1)

      inputs = Map.put(inputs, "seed", seed)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, {multi?, input_length, input_padded_length}}
    end)
    |> maybe_stream(opts[:stream], opts[:stream_done], tokenizer)
  end

  defp validate_input(text) when is_binary(text), do: validate_input(%{text: text})

  defp validate_input(%{text: text} = input) do
    {:ok, %{text: text, seed: input[:seed] || :erlang.system_time()}}
  end

  defp validate_input(%{} = input) do
    {:error, "expected the input map to have :text key, got: #{inspect(input)}"}
  end

  defp validate_input(input) do
    {:error, "expected either a string or a map, got: #{inspect(input)}"}
  end

  defp maybe_stream(serving, false, _stream_done, tokenizer) do
    Nx.Serving.client_postprocessing(
      serving,
      fn {%{token_ids: token_ids, length: length}, _metadata},
         {multi?, input_length, input_padded_length} ->
        decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)
        output_length = Nx.to_flat_list(length)
        input_length = Nx.to_flat_list(input_length)

        Enum.zip_with(
          [decoded, output_length, input_length],
          fn [decoded, output_length, input_length] ->
            token_summary = token_summary(input_length, input_padded_length, output_length)
            %{results: [%{text: decoded, token_summary: token_summary}]}
          end
        )
        |> Shared.normalize_output(multi?)
      end
    )
  end

  defp maybe_stream(serving, true, stream_done, tokenizer) do
    serving
    |> Nx.Serving.streaming(hooks: [:token])
    |> Nx.Serving.client_postprocessing(fn stream,
                                           {false = _multi?, input_length, input_padded_length} ->
      [input_length] = Nx.to_flat_list(input_length)

      Stream.transform(stream, %{tokens: [], consumed_size: 0, finished?: false}, fn
        _event, %{finished?: true} = state ->
          {:halt, state}

        {:token, %{token_id: token_id, finished?: finished?, length: output_length}}, state ->
          token_id = Nx.to_number(token_id[0])
          finished? = Nx.to_number(finished?[0]) == 1

          state = %{state | tokens: state.tokens ++ [token_id], finished?: finished?}

          chunk = pending_chunk(tokenizer, state)

          {items, state} =
            cond do
              # When the sequence is finished early or we reach a newline,
              # we flush the cache
              finished? or String.ends_with?(chunk, "\n") ->
                {[chunk], %{state | tokens: [], consumed_size: 0}}

              # CJK characters are tokenized atomically, so we can emit
              # the chunk
              chunk != "" and cjk_codepoint?(last_codepoint(chunk)) ->
                state = update_in(state.consumed_size, &(&1 + byte_size(chunk)))
                {[chunk], state}

              # Emit chunk until the space. We need to keep tokens,
              # because certain tokenizers do not encode whitespace in
              # tokens and they add a space based on previous tokens
              space_idx = find_last_occurrence(chunk, " ") ->
                if space_idx > 0 do
                  chunk = binary_slice(chunk, 0, space_idx)
                  state = update_in(state.consumed_size, &(&1 + space_idx))
                  {[chunk], state}
                else
                  {[], state}
                end

              true ->
                {[], state}
            end

          if finished? and stream_done do
            output_length = Nx.to_number(output_length[0])
            token_summary = token_summary(input_length, input_padded_length, output_length)
            done = {:done, %{token_summary: token_summary}}
            {items ++ [done], state}
          else
            {items, state}
          end
      end)
    end)
  end

  defp token_summary(input_length, input_padded_length, output_length) do
    %{
      input: input_length,
      output: output_length,
      padding: input_padded_length - input_length
    }
  end

  defp pending_chunk(tokenizer, state) do
    text = Bumblebee.Tokenizer.decode(tokenizer, state.tokens)
    binary_slice(text, state.consumed_size..-1//1)
  end

  defp find_last_occurrence(string, pattern) do
    case :binary.matches(string, pattern) do
      [] -> nil
      matches -> matches |> List.last() |> elem(0)
    end
  end

  defp last_codepoint(<<codepoint::utf8>>), do: codepoint
  defp last_codepoint(<<_::utf8, rest::binary>>), do: last_codepoint(rest)

  defp cjk_codepoint?(codepoint) do
    # The specific ranges originated in [1] and are generally mirrored
    # in other tokenizers using WordPiece. Also see [2].
    #
    # [1]: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L264-L284
    # [2]: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/multilingual.md#tokenization

    codepoint in 0x4E00..0x9FFF or
      codepoint in 0x3400..0x4DBF or
      codepoint in 0x20000..0x2A6DF or
      codepoint in 0x2A700..0x2B73F or
      codepoint in 0x2B740..0x2B81F or
      codepoint in 0x2B820..0x2CEAF or
      codepoint in 0xF900..0xFAFF or
      codepoint in 0x2F800..0x2FA1F
  end
end
