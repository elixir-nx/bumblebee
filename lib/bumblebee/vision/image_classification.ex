defmodule Bumblebee.Vision.ImageClassification do
  @moduledoc false

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  def image_classification(model_info, featurizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [
      :for_image_classification,
      :for_image_classification_with_teacher
    ])

    opts = Keyword.validate!(opts, [:compile, top_k: 5, defn_options: []])

    top_k = opts[:top_k]
    compile = opts[:compile]
    defn_options = opts[:defn_options]

    batch_size = compile[:batch_size]

    if compile != nil and batch_size == nil do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size, got: #{inspect(compile)}"
    end

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      outputs = predict_fun.(params, input)
      Axon.Activations.softmax(outputs.logits)
    end

    Nx.Serving.new(
      fn defn_options ->
        scores_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "pixel_values" => Shared.input_template(spec, "pixel_values", [batch_size])
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          scores_fun.(params, inputs)
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
    |> Nx.Serving.client_postprocessing(fn scores, _metadata, multi? ->
      for scores <- Bumblebee.Utils.Nx.batch_to_list(scores) do
        k = min(top_k, Nx.size(scores))
        {top_scores, top_indices} = Utils.Nx.top_k(scores, k: k)

        predictions =
          Enum.zip_with(
            Nx.to_flat_list(top_scores),
            Nx.to_flat_list(top_indices),
            fn score, idx ->
              label = spec.id_to_label[idx] || "LABEL_#{idx}"
              %{score: score, label: label}
            end
          )

        %{predictions: predictions}
      end
      |> Shared.normalize_output(multi?)
    end)
  end
end
