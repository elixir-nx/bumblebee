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
        stream_done: false,
        include_timing: false,
        output_format: :bumblebee,
        model_name: nil
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

        scope = {:generate, batch_key}

        generate_fun =
          Shared.compile_or_jit(generate_fun, scope, defn_options, compile != nil, fn ->
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

      start_time =
        if opts[:include_timing], do: System.monotonic_time(:microsecond), else: nil

      {inputs, multi?} = Shared.validate_serving_input!(input, &validate_input/1)

      texts = Enum.map(inputs, & &1.text)
      seed = Enum.map(inputs, & &1.seed) |> Nx.tensor(type: :s64, backend: Nx.BinaryBackend)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts)
        end)

      {input_length, inputs} = Map.pop!(inputs, "length")
      input_padded_length = Nx.axis_size(inputs["input_ids"], 1)

      inputs = Map.put(inputs, "seed", seed)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, {multi?, input_length, input_padded_length, start_time}}
    end)
    |> add_postprocessing(tokenizer,
      stream: opts[:stream],
      stream_done: opts[:stream_done],
      include_timing: opts[:include_timing],
      output_format: opts[:output_format],
      model_name: opts[:model_name]
    )
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

  @doc false
  def add_postprocessing(serving, tokenizer, opts) do
    if opts[:stream] do
      add_streaming_postprocessing(serving, tokenizer, opts)
    else
      add_non_streaming_postprocessing(serving, tokenizer, opts)
    end
  end

  defp add_non_streaming_postprocessing(serving, tokenizer, opts) do
    include_timing = opts[:include_timing]
    output_format = opts[:output_format] || :bumblebee
    model_name = opts[:model_name]

    Nx.Serving.client_postprocessing(
      serving,
      fn {%{token_ids: token_ids, length: length, finish_reason: finish_reason}, _metadata},
         {multi?, input_length, input_padded_length, start_time} ->
        end_time =
          if include_timing, do: System.monotonic_time(:microsecond), else: nil

        decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)
        output_length_list = Nx.to_flat_list(length)
        input_length_list = Nx.to_flat_list(input_length)
        finish_reason_list = Nx.to_flat_list(finish_reason)

        results =
          Enum.zip_with(
            [decoded, output_length_list, input_length_list],
            fn [decoded, output_length, input_length] ->
              token_summary = token_summary(input_length, input_padded_length, output_length)
              result = %{text: decoded, token_summary: token_summary}

              if include_timing && start_time do
                duration_us = end_time - start_time
                tokens_per_second = output_length / (duration_us / 1_000_000)

                Map.merge(result, %{
                  generation_time_us: duration_us,
                  tokens_per_second: tokens_per_second
                })
              else
                result
              end
            end
          )

        timing =
          if include_timing && start_time do
            total_output = Enum.sum(output_length_list)
            duration_us = end_time - start_time

            %{
              duration_us: duration_us,
              tokens_per_second: total_output / (duration_us / 1_000_000)
            }
          else
            nil
          end

        format_output(results, finish_reason_list, output_format, model_name, timing, multi?)
      end
    )
  end

  defp add_streaming_postprocessing(serving, tokenizer, opts) do
    stream_done = opts[:stream_done]
    include_timing = opts[:include_timing]
    output_format = opts[:output_format] || :bumblebee
    model_name = opts[:model_name]

    serving
    |> Nx.Serving.streaming(hooks: [:token])
    |> Nx.Serving.client_postprocessing(fn stream,
                                           {false = _multi?, input_length, input_padded_length,
                                            start_time} ->
      [input_length] = Nx.to_flat_list(input_length)

      # Generate stream ID for OpenAI formats
      stream_id = generate_stream_id(output_format)

      initial_state = %{
        tokens: [],
        consumed_size: 0,
        finished?: false,
        first_token_time: nil,
        stream_id: stream_id
      }

      Stream.transform(stream, initial_state, fn
        _event, %{finished?: true} = state ->
          {:halt, state}

        {:token,
         %{
           token_id: token_id,
           finished?: finished?,
           length: output_length,
           finish_reason: finish_reason
         }},
        state ->
          token_id = Nx.to_number(token_id[0])
          finished? = Nx.to_number(finished?[0]) == 1
          finish_reason_val = Nx.to_number(finish_reason[0])

          # Track first token time
          first_token_time =
            if state.first_token_time == nil && include_timing do
              System.monotonic_time(:microsecond)
            else
              state.first_token_time
            end

          state = %{
            state
            | tokens: state.tokens ++ [token_id],
              finished?: finished?,
              first_token_time: first_token_time
          }

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

          # Format items for OpenAI streaming if needed
          items = format_stream_chunks(items, output_format, state.stream_id, model_name)

          if finished? and stream_done do
            output_length = Nx.to_number(output_length[0])
            token_summary = token_summary(input_length, input_padded_length, output_length)

            done_result = %{
              token_summary: token_summary,
              finish_reason: map_finish_reason(finish_reason_val)
            }

            done_result =
              if include_timing && start_time do
                end_time = System.monotonic_time(:microsecond)
                duration_us = end_time - start_time

                time_to_first_token_us =
                  if state.first_token_time,
                    do: state.first_token_time - start_time,
                    else: nil

                tokens_per_second = output_length / (duration_us / 1_000_000)

                Map.merge(done_result, %{
                  generation_time_us: duration_us,
                  time_to_first_token_us: time_to_first_token_us,
                  tokens_per_second: tokens_per_second
                })
              else
                done_result
              end

            done = {:done, done_result}
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

  # Output format helpers

  defp format_output(results, _finish_reasons, :bumblebee, _model_name, _timing, multi?) do
    results
    |> Enum.map(fn result -> %{results: [result]} end)
    |> Shared.normalize_output(multi?)
  end

  defp format_output(results, finish_reasons, :openai, model_name, timing, _multi?) do
    to_openai_completion_format(results, model_name, finish_reasons, timing)
  end

  defp format_output(results, finish_reasons, :openai_chat, model_name, timing, _multi?) do
    to_openai_chat_format(results, model_name, finish_reasons, timing)
  end

  defp to_openai_completion_format(results, model_name, finish_reasons, timing) do
    %{
      id: "cmpl-" <> Base.encode16(:crypto.strong_rand_bytes(12), case: :lower),
      object: "text_completion",
      created: System.system_time(:second),
      model: model_name || "unknown",
      choices:
        Enum.with_index(results, fn result, i ->
          %{
            index: i,
            text: result.text,
            finish_reason: map_finish_reason(Enum.at(finish_reasons, i))
          }
        end),
      usage: build_usage(results, timing)
    }
  end

  defp to_openai_chat_format(results, model_name, finish_reasons, timing) do
    %{
      id: "chatcmpl-" <> Base.encode16(:crypto.strong_rand_bytes(12), case: :lower),
      object: "chat.completion",
      created: System.system_time(:second),
      model: model_name || "unknown",
      choices:
        Enum.with_index(results, fn result, i ->
          %{
            index: i,
            message: %{
              role: "assistant",
              content: result.text
            },
            finish_reason: map_finish_reason(Enum.at(finish_reasons, i))
          }
        end),
      usage: build_usage(results, timing)
    }
  end

  defp build_usage(results, timing) do
    usage = %{
      prompt_tokens: Enum.sum(Enum.map(results, & &1.token_summary.input)),
      completion_tokens: Enum.sum(Enum.map(results, & &1.token_summary.output)),
      total_tokens:
        Enum.sum(Enum.map(results, &(&1.token_summary.input + &1.token_summary.output)))
    }

    if timing do
      Map.merge(usage, %{
        generation_time_us: timing.duration_us,
        tokens_per_second: timing.tokens_per_second
      })
    else
      usage
    end
  end

  defp map_finish_reason(1), do: "stop"
  defp map_finish_reason(2), do: "length"
  defp map_finish_reason(_), do: nil

  # Streaming format helpers

  defp generate_stream_id(:openai),
    do: "cmpl-" <> Base.encode16(:crypto.strong_rand_bytes(12), case: :lower)

  defp generate_stream_id(:openai_chat),
    do: "chatcmpl-" <> Base.encode16(:crypto.strong_rand_bytes(12), case: :lower)

  defp generate_stream_id(_), do: nil

  defp format_stream_chunks(chunks, :bumblebee, _stream_id, _model_name), do: chunks

  defp format_stream_chunks(chunks, :openai, stream_id, model_name) do
    Enum.map(chunks, fn chunk ->
      %{
        id: stream_id,
        object: "text_completion",
        created: System.system_time(:second),
        model: model_name || "unknown",
        choices: [%{index: 0, text: chunk, finish_reason: nil}]
      }
    end)
  end

  defp format_stream_chunks(chunks, :openai_chat, stream_id, model_name) do
    Enum.map(chunks, fn chunk ->
      %{
        id: stream_id,
        object: "chat.completion.chunk",
        created: System.system_time(:second),
        model: model_name || "unknown",
        choices: [%{index: 0, delta: %{content: chunk}, finish_reason: nil}]
      }
    end)
  end
end
