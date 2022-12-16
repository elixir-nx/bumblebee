defmodule Bumblebee.Text.ZeroShotClassification do
  @moduledoc false

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  def zero_shot_classification(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_sequence_classification)

    opts =
      Keyword.validate!(opts, [
        :aggregation,
        :compile,
        ignored_labels: ["O"],
        defn_options: []
      ])

    compile = opts[:compile]
    defn_options = opts[:defn_options]
    hypothesis_template = opts[:hypothesis_template] || (&default_hypothesis_template/1)

    batch_size = compile[:batch_size]
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
          &validate_input/1,
          "a map of %{prompt: prompt, labels: labels}"
        )

      {[prompt | _] = prompts, labels_and_hypothesis} =
        texts
        |> get_inputs(hypothesis_template)
        |> Enum.unzip()

      {labels, hypothesis} = Enum.unzip(labels_and_hypothesis)

      all_inputs =
        Bumblebee.apply_tokenizer(tokenizer, Enum.zip(prompts, hypothesis),
          length: sequence_length,
          return_special_tokens_mask: true,
          return_offsets: true
        )

      inputs = Map.take(all_inputs, ["input_ids", "attention_mask"])

      {Nx.Batch.concatenate([inputs]), {[prompt], labels, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn scores, _metadata, {[prompt], labels, multi?} ->
      # TODO: Handle the case where scores comes from multiple prompts
      scores = Nx.to_flat_list(scores[[0..-1//1, 1]])

      [%{scores: scores, labels: labels, prompt: prompt}]
      |> Shared.normalize_output(multi?)
    end)
  end

  defp default_hypothesis_template(label), do: "This example is #{label}"

  defp validate_input(%{prompt: prompt, labels: labels})
       when is_binary(prompt) and is_list(labels) do
    Enum.all?(labels, &is_binary/1)
  end

  defp get_inputs(%{prompt: prompt, labels: labels}, hypothesis_template)
       when is_binary(prompt) and is_list(labels) do
    Enum.map(labels, fn lab -> {prompt, {lab, hypothesis_template.(lab)}} end)
  end

  defp get_inputs(inputs, hypothesis_template) when is_list(inputs),
    do: Enum.flat_map(inputs, &get_inputs(&1, hypothesis_template))
end
