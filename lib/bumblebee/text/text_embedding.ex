defmodule Bumblebee.Text.TextEmbedding do
  @moduledoc false

  alias Bumblebee.Shared

  def text_embedding(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: _spec} = model_info

    opts = Keyword.validate!(opts, [:compile, :output_attribute, defn_options: []])

    compile = opts[:compile]
    output_attribute = opts[:output_attribute]
    defn_options = opts[:defn_options]

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    {_init_fun, encoder} = Axon.build(model)

    embedding_fun = fn params, input ->
      encoder.(params, input)[output_attribute]
    end

    Nx.Serving.new(
      fn defn_options ->
        embedding_fun =
          Shared.compile_or_jit(embedding_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          embedding_fun.(params, inputs)
        end
      end,
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, texts,
          length: sequence_length,
          return_token_type_ids: false
        )

      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn embeddings, _metadata, multi? ->
      for embedding <- Bumblebee.Utils.Nx.batch_to_list(embeddings) do
        %{embedding: embedding}
      end
      |> Shared.normalize_output(multi?)
    end)
  end
end
