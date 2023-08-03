defmodule Bumblebee.Vision.ImageEmbedding do
  @moduledoc false

  alias Bumblebee.Shared

  def image_embedding(model_info, featurizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info

    opts =
      Keyword.validate!(opts, [
        :compile,
        output_attribute: :pooled_state,
        embedding_processor: nil,
        defn_options: [],
        preallocate_params: false
      ])

    output_attribute = opts[:output_attribute]
    embedding_processor = opts[:embedding_processor]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size])
        |> Shared.require_options!([:batch_size])
      end

    batch_size = compile[:batch_size]

    {_init_fun, encoder} = Axon.build(model)

    embedding_fun = fn params, inputs ->
      output = encoder.(params, inputs)

      output =
        if is_map(output) do
          output[output_attribute]
        else
          output
        end

      output =
        case embedding_processor do
          nil ->
            output

          :l2_norm ->
            Bumblebee.Utils.Nx.normalize(output)

          other ->
            raise ArgumentError,
                  "expected :embedding_processor to be one of nil or :l2_norm, got: #{inspect(other)}"
        end

      output
    end

    Nx.Serving.new(
      fn defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        embedding_fun =
          Shared.compile_or_jit(embedding_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "pixel_values" => Shared.input_template(spec, "pixel_values", [batch_size])
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
      {images, multi?} = Shared.validate_serving_input!(input, &Shared.validate_image/1)

      inputs = Bumblebee.apply_featurizer(featurizer, images)

      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {embeddings, _metadata}, multi? ->
      for embedding <- Bumblebee.Utils.Nx.batch_to_list(embeddings) do
        %{embedding: embedding}
      end
      |> Shared.normalize_output(multi?)
    end)
  end
end
