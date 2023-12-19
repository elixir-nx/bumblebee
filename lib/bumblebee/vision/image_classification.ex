defmodule Bumblebee.Vision.ImageClassification do
  @moduledoc false

  alias Bumblebee.Shared

  def image_classification(model_info, featurizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [
      :for_image_classification,
      :for_image_classification_with_teacher
    ])

    opts =
      Keyword.validate!(opts, [
        :compile,
        top_k: 5,
        scores_function: :softmax,
        defn_options: [],
        preallocate_params: false
      ])

    top_k = opts[:top_k]
    scores_function = opts[:scores_function]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size])
        |> Shared.require_options!([:batch_size])
      end

    batch_size = compile[:batch_size]

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      input = Bumblebee.Featurizer.process_batch(featurizer, input)
      outputs = predict_fun.(params, input)
      scores = Shared.logits_to_scores(outputs.logits, scores_function)
      k = min(top_k, Nx.axis_size(scores, 1))
      {top_scores, top_indices} = Nx.top_k(scores, k: k)
      {top_scores, top_indices}
    end

    Nx.Serving.new(
      fn defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        scores_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            inputs = Bumblebee.Featurizer.batch_template(featurizer, batch_size)
            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          scores_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {images, multi?} = Shared.validate_serving_input!(input, &Shared.validate_image/1)
      inputs = Bumblebee.Featurizer.process_input(featurizer, images)
      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {{top_scores, top_indices}, _metadata}, multi? ->
      Enum.zip_with(
        Nx.to_list(top_scores),
        Nx.to_list(top_indices),
        fn top_scores, top_indices ->
          predictions =
            Enum.zip_with(top_scores, top_indices, fn score, idx ->
              label = spec.id_to_label[idx] || "LABEL_#{idx}"
              %{score: score, label: label}
            end)

          %{predictions: predictions}
        end
      )
      |> Shared.normalize_output(multi?)
    end)
  end
end
