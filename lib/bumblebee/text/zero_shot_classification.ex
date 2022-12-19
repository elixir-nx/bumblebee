defmodule Bumblebee.Text.ZeroShotClassification do
  @moduledoc false

  alias Bumblebee.Shared

  def zero_shot_classification(model_info, tokenizer, labels, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_sequence_classification)

    opts =
      Keyword.validate!(opts, [
        :compile,
        hypothesis_template: &default_hypothesis_template/1,
        defn_options: []
      ])

    compile = opts[:compile]
    defn_options = opts[:defn_options]
    hypothesis_template = opts[:hypothesis_template] || (&default_hypothesis_template/1)

    hypotheses = Enum.map(labels, hypothesis_template)

    sequences_per_batch = length(labels)

    batch_size =
      if batch_size = compile[:batch_size] do
        batch_size * sequences_per_batch
      else
        nil
      end

    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      %{logits: logits} = predict_fun.(params, input)
      logits = Nx.take(logits, Nx.tensor([0, 2]), axis: 1)
      Axon.Activations.softmax(logits)
    end

    Nx.Serving.new(
      fn ->
        scores_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :s64),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :s64)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          scores_fun.(params, inputs)
        end
      end,
      batch_size: batch_size
    )
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} =
        Shared.validate_serving_input!(
          input,
          &is_binary/1,
          "a binary"
        )

      pairs = Enum.flat_map(texts, fn text -> Enum.map(hypotheses, fn hyp -> {text, hyp} end) end)

      all_inputs =
        Bumblebee.apply_tokenizer(tokenizer, pairs,
          length: sequence_length,
          return_special_tokens_mask: true,
          return_offsets: true
        )

      inputs = Map.take(all_inputs, ["input_ids", "attention_mask"])

      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn scores, _metadata, multi? ->
      scores
      |> Nx.to_batched(sequences_per_batch)
      |> Enum.map(fn scores_for_batch ->
        scores_for_batch[[0..-1//1, 1]]
        |> Nx.to_flat_list()
        |> Enum.zip_with(labels, fn score, label -> %{score: score, label: label} end)
      end)
      |> Shared.normalize_output(multi?)
    end)
  end

  defp default_hypothesis_template(label), do: "This example is #{label}."
end
