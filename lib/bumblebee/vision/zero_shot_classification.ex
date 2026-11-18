defmodule Bumblebee.Vision.ZeroShotClassification do
  @moduledoc false

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  def zero_shot_classification(model_info, featurizer, tokenizer, labels, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :base)

    opts =
      Keyword.validate!(opts, [
        :compile,
        hypothesis_template: &default_hypothesis_template/1,
        defn_options: [],
        preallocate_params: false
      ])

    hypothesis_template = opts[:hypothesis_template]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    hypotheses = Enum.map(labels, hypothesis_template)

    tokenized_hypotheses =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Bumblebee.apply_tokenizer(tokenizer, hypotheses, return_token_type_ids: false)
      end)

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size])
        |> Shared.require_options!([:batch_size])
      end

    batch_size = compile[:batch_size]

    {_init_fun, predict_fun} = Axon.build(model)

    logits_fun = fn params, input ->
      input =
        featurizer
        |> Bumblebee.Featurizer.process_batch(input)
        |> Map.merge(tokenized_hypotheses)

      %{logits_per_image: logits_per_image} = predict_fun.(params, input)

      logits_per_image
    end

    Nx.Serving.new(
      fn defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        logits_fun =
          Shared.compile_or_jit(logits_fun, defn_options, compile != nil, fn ->
            inputs = Bumblebee.Featurizer.batch_template(featurizer, batch_size)
            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          logits = logits_fun.(params, inputs)
          Axon.Activations.softmax(logits)
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
    |> Nx.Serving.client_postprocessing(fn {scores, _metadata}, multi? ->
      scores
      |> Utils.Nx.to_list()
      |> Enum.map(fn scores_for_batch ->
        Enum.zip_with(scores_for_batch, labels, fn score, label ->
          %{score: score, label: label}
        end)
      end)
      |> Shared.normalize_output(multi?)
    end)
  end

  defp default_hypothesis_template(label), do: "This is a photo of #{label}."
end
