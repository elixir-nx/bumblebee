defmodule Bumblebee.Text.Conversation do
  @moduledoc false

  alias Bumblebee.Shared
  alias Bumblebee.Text

  @doc """
  Converts conversation history into a continuous text.
  """
  @callback conversation_history_to_text(
              Bumblebee.Tokenizer.t(),
              Bumblebee.Text.conversational_history()
            ) :: String.t()

  @doc false
  def conversation(
        model_info,
        tokenizer,
        %Text.GenerationConfig{} = generation_config,
        opts \\ []
      ) do
    opts = Keyword.validate!(opts, [:compile, defn_options: [], preallocate_params: false])

    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [
      :for_causal_language_modeling,
      :for_conditional_generation
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

    generate_fun = Text.Generation.build_generate(model, spec, generation_config)

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
      {inputs, multi?} = Shared.validate_serving_input!(input, &validate_input/1)

      histories = Enum.map(inputs, & &1.history)
      seed = Enum.map(inputs, & &1.seed) |> Nx.tensor(backend: Nx.BinaryBackend)

      texts =
        for history <- histories do
          tokenizer.__struct__.conversation_history_to_text(tokenizer, history)
        end

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts,
            length: sequence_length,
            pad_direction: :left,
            truncate_direction: :left,
            return_token_type_ids: false
          )
        end)

      inputs = Map.put(inputs, "seed", seed)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, {histories, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn {token_ids, _metadata}, {histories, multi?} ->
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      Enum.zip_with(decoded, histories, fn text, history ->
        %{text: text, history: [{:generated, text} | history]}
      end)
      |> Shared.normalize_output(multi?)
    end)
  end

  defp validate_input(%{text: text, history: history} = input) when is_binary(text) do
    history = history || []
    history = [{:user, text} | history]
    {:ok, %{history: history, seed: input[:seed] || :erlang.system_time()}}
  end

  defp validate_input(input) do
    {:error, "expected input to be a map with :text and :history, got: #{inspect(input)}"}
  end
end
