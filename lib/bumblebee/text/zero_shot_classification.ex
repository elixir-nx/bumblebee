defmodule Bumblebee.Text.ZeroShotClassification do
  @moduledoc false

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  def zero_shot_classification(model_info, tokenizer, labels, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_sequence_classification)

    opts =
      Keyword.validate!(opts, [
        :compile,
        hypothesis_template: &default_hypothesis_template/1,
        top_k: 5,
        defn_options: []
      ])

    hypothesis_template = opts[:hypothesis_template]
    top_k = opts[:top_k]
    defn_options = opts[:defn_options]

    hypotheses = Enum.map(labels, hypothesis_template)

    sequences_per_batch = length(labels)

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size, :sequence_length])
        |> Shared.require_options!([:batch_size, :sequence_length])
      end

    sequence_length = compile[:sequence_length]
    batch_size = compile[:batch_size]

    entailment_id =
      Enum.find_value(spec.id_to_label, fn {id, label} ->
        label == "entailment" && id
      end)

    unless entailment_id do
      raise ArgumentError,
            ~s/expected model specification to include "entailment" label in :id_to_label/
    end

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      input = Utils.Nx.composite_flatten_batch(input)
      %{logits: logits} = predict_fun.(params, input)
      logits
    end

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        scores_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" =>
                Nx.template({batch_size, sequences_per_batch, sequence_length}, :u32),
              "attention_mask" =>
                Nx.template({batch_size, sequences_per_batch, sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          scores = scores_fun.(params, inputs)
          Utils.Nx.composite_unflatten_batch(scores, inputs.size)
        end
      end,
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size, batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      pairs = for text <- texts, hypothesis <- hypotheses, do: {text, hypothesis}

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, pairs,
          length: sequence_length,
          return_token_type_ids: false
        )

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)

      inputs = Utils.Nx.composite_unflatten_batch(inputs, length(texts))

      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {scores, _metadata}, multi? ->
      for scores <- Utils.Nx.batch_to_list(scores) do
        scores = Axon.Activations.softmax(scores[[.., entailment_id]])

        k = min(top_k, Nx.size(scores))
        {top_scores, top_indices} = Nx.top_k(scores, k: k)

        predictions =
          Enum.zip_with(
            Nx.to_flat_list(top_scores),
            Nx.to_flat_list(top_indices),
            fn score, idx ->
              label = Enum.at(labels, idx)
              %{score: score, label: label}
            end
          )

        %{predictions: predictions}
      end
      |> Shared.normalize_output(multi?)
    end)
  end

  defp default_hypothesis_template(label), do: "This example is #{label}."
end
