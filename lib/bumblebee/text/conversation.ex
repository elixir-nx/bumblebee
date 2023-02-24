defmodule Bumblebee.Text.Conversation do
  @moduledoc false

  alias Bumblebee.Shared

  @doc """
  Converts conversation history into a continuous text.
  """
  @callback conversation_history_to_text(
              Bumblebee.Tokenizer.t(),
              Bumblebee.Text.conversational_history()
            ) :: String.t()

  @doc false
  def conversation(model_info, tokenizer, opts \\ []) do
    %{params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [:for_causal_language_modeling])

    {compile, opts} = Keyword.pop(opts, :compile)
    {defn_options, opts} = Keyword.pop(opts, :defn_options, [])

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    generate_fun =
      Bumblebee.Text.Generation.build_generate(model_info.model, model_info.spec, opts)

    Nx.Serving.new(
      fn defn_options ->
        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :s64),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :s64)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          sequences = generate_fun.(params, inputs)
          inputs = Nx.Defn.jit_apply(&Function.identity/1, [inputs])
          input_length = Nx.axis_size(inputs["input_ids"], 1)
          sequences[[0..-1//1, input_length..-1//1]]
        end
      end,
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {histories, multi?} = Shared.validate_serving_input!(input, &validate_input/1)

      texts =
        for history <- histories do
          tokenizer.__struct__.conversation_history_to_text(tokenizer, history)
        end

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, texts,
          length: sequence_length,
          pad_direction: :left,
          truncate_direction: :left,
          return_token_type_ids: false
        )

      {Nx.Batch.concatenate([inputs]), {histories, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn token_ids, _metadata, {histories, multi?} ->
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      Enum.zip_with(decoded, histories, fn text, history ->
        %{text: text, history: [{:generated, text} | history]}
      end)
      |> Shared.normalize_output(multi?)
    end)
  end

  defp validate_input(%{text: text, history: history}) when is_binary(text) do
    history = history || []
    {:ok, [{:user, text} | history]}
  end

  defp validate_input(input) do
    {:error, "expected input to be a map with :text and :history, got: #{inspect(input)}"}
  end
end
