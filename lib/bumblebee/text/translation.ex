defmodule Bumblebee.Text.Translation do
  @moduledoc false

  alias Bumblebee.Shared
  alias Bumblebee.Text

  def translation(model_info, tokenizer, %Text.GenerationConfig{} = generation_config, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :compile,
        defn_options: [],
        preallocate_params: false,
        stream: false,
        stream_done: false
      ])

    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [:for_conditional_generation])

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
              "decoder_input_ids" => Nx.template({batch_size, 2}, :u32),
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
      seed = Enum.map(inputs, & &1.seed) |> Nx.tensor(type: :s64, backend: Nx.BinaryBackend)

      source_language_token = source_language_token!(inputs)

      validate_language_token!(source_language_token, tokenizer)

      tokenizer =
        Bumblebee.configure(tokenizer, template_options: [language_token: source_language_token])

      # We specify custom decoder_input_ids input to include the dynamic
      # language token id after the start token
      decoder_input_ids =
        inputs
        |> Enum.map(fn %{target_language_token: target_language_token} ->
          validate_language_token!(target_language_token, tokenizer)
          token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, target_language_token)

          decoder_start_token_id =
            generation_config.decoder_start_token_id || generation_config.bos_token_id

          [decoder_start_token_id, token_id]
        end)
        |> Nx.tensor(type: :u32, backend: Nx.BinaryBackend)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts)
        end)

      {input_length, inputs} = Map.pop!(inputs, "length")
      input_padded_length = Nx.axis_size(inputs["input_ids"], 1)

      inputs = Map.put(inputs, "seed", seed)
      inputs = Map.put(inputs, "decoder_input_ids", decoder_input_ids)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, {multi?, input_length, input_padded_length}}
    end)
    |> Text.TextGeneration.add_postprocessing(opts[:stream], opts[:stream_done], tokenizer)
  end

  defp validate_input(
         %{
           text: text,
           source_language_token: source_language_token,
           target_language_token: target_language_token
         } = input
       ) do
    {:ok,
     %{
       text: text,
       source_language_token: source_language_token,
       target_language_token: target_language_token,
       seed: input[:seed] || :erlang.system_time()
     }}
  end

  defp validate_input(%{} = input) do
    {:error,
     "expected the input map to have :text, :source_language_token and :target_language_token keys, got: #{inspect(input)}"}
  end

  defp validate_input(input) do
    {:error, "expected a map, got: #{inspect(input)}"}
  end

  defp source_language_token!(inputs) do
    source_language_tokens = for input <- inputs, uniq: true, do: input.source_language_token

    case source_language_tokens do
      [token] ->
        token

      _tokens ->
        raise ArgumentError,
              "the translation serving supports a list of inputs only when all" <>
                " of them have the same :source_language_token. To process multiple" <>
                " inputs with different source language, configure :compile options" <>
                " with a desired batch size, start a serving process and use" <>
                " Task.async_stream/1 with Nx.Serving.batched_run/1"
    end
  end

  defp validate_language_token!(language_token, tokenizer) do
    unless Bumblebee.Tokenizer.token_to_id(tokenizer, language_token) do
      raise ArgumentError,
            "the specified language token #{inspect(language_token)} is not" <>
              " a valid token for this tokenizer"
    end
  end
end
