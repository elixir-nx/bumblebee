defmodule Bumblebee.Text.TokenClassification do
  @moduledoc false

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  def token_classification(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_token_classification)

    opts =
      Keyword.validate!(opts, [
        :aggregation,
        :compile,
        ignored_labels: ["O"],
        defn_options: []
      ])

    aggregation = opts[:aggregation]
    ignored_labels = opts[:ignored_labels]
    compile = opts[:compile]
    defn_options = opts[:defn_options]

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      outputs = predict_fun.(params, input)
      Axon.Activations.softmax(outputs.logits)
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
      {texts, multi?} = Shared.validate_serving_input!(input, &is_binary/1, "a string")

      all_inputs =
        Bumblebee.apply_tokenizer(tokenizer, texts,
          length: sequence_length,
          return_special_tokens_mask: true,
          return_offsets: true
        )

      inputs = Map.take(all_inputs, ["input_ids", "attention_mask"])

      {Nx.Batch.concatenate([inputs]), {all_inputs, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn scores, _metadata, {inputs, multi?} ->
      Enum.zip_with(
        Utils.Nx.batch_to_list(inputs),
        Utils.Nx.batch_to_list(scores),
        fn inputs, scores ->
          entities =
            scores
            |> gather_raw_entities(tokenizer, inputs)
            |> aggregate(spec, tokenizer, aggregation)
            |> filter_entities(ignored_labels)

          %{entities: entities}
        end
      )
      |> Shared.normalize_output(multi?)
    end)
  end

  defp gather_raw_entities(scores, tokenizer, inputs) do
    {sequence_length, _} = Nx.shape(scores)
    flat_special_tokens_mask = Nx.to_flat_list(inputs["special_tokens_mask"])
    flat_input_ids = Nx.to_flat_list(inputs["input_ids"])
    flat_start_offsets = Nx.to_flat_list(inputs["start_offsets"])
    flat_end_offsets = Nx.to_flat_list(inputs["end_offsets"])

    # TODO: Optional offset mapping
    # TODO: Non-BPE tokenizers
    token_infos =
      Enum.zip([
        0..(sequence_length - 1),
        flat_input_ids,
        flat_start_offsets,
        flat_end_offsets,
        flat_special_tokens_mask
      ])

    for {token_idx, token_id, start_idx, end_idx, _special? = 0} <- token_infos do
      token = Bumblebee.Tokenizer.id_to_token(tokenizer, token_id)
      # Indices are expressed in terms of utf8 bytes
      token_reference_length = end_idx - start_idx

      token_scores = scores[token_idx]

      %{
        token: token,
        token_id: token_id,
        scores: token_scores,
        start: start_idx,
        end: end_idx,
        index: token_idx,
        # Subword tokens usually have the ## prefix, so they are longer
        # than the actual word piece
        is_subword: byte_size(token) != token_reference_length
      }
    end
  end

  defp aggregate(entities, spec, _tokenizer, nil) do
    entities
    |> add_token_labels(spec)
    |> Enum.map(fn entity ->
      %{
        start: entity.start,
        end: entity.end,
        label: entity.label,
        score: entity.score,
        phrase: entity.token
      }
    end)
  end

  defp aggregate(entities, spec, tokenizer, :same) do
    entities
    |> add_token_labels(spec)
    |> group_entities(tokenizer)
  end

  defp filter_entities(entities, ignored_labels) do
    Enum.filter(entities, fn entity ->
      {_prefix, label} = parse_label(entity.label)
      label not in ignored_labels
    end)
  end

  defp add_token_labels(entities, spec) do
    Enum.map(entities, fn entity ->
      entity_idx = entity.scores |> Nx.argmax() |> Nx.to_number()
      score = Nx.to_number(entity.scores[entity_idx])
      label = spec.id_to_label[entity_idx]
      Map.merge(entity, %{label: label, score: score})
    end)
  end

  defp group_entities([entity | entities], tokenizer) do
    {_prefix, label} = parse_label(entity.label)

    {groups, _} =
      Enum.reduce(entities, {[[entity]], label}, fn entity, {[group | groups], prev_label} ->
        case parse_label(entity.label) do
          {:i, ^prev_label} ->
            {[[entity | group] | groups], prev_label}

          {_, label} ->
            {[[entity], group | groups], label}
        end
      end)

    groups
    |> Enum.map(&finish_group(&1, tokenizer))
    |> Enum.reverse()
  end

  defp finish_group([last_entity | _] = rev_group, tokenizer) do
    [first_entity | _] = group = Enum.reverse(rev_group)
    {_, label} = parse_label(first_entity.label)
    scores = Enum.map(group, & &1.score)
    token_ids = Enum.map(group, & &1.token_id)

    %{
      start: first_entity.start,
      end: last_entity.end,
      label: label,
      score: Enum.sum(scores) / length(scores),
      phrase: Bumblebee.Tokenizer.decode(tokenizer, token_ids)
    }
  end

  # Parse the BIO tagging format
  defp parse_label("B-" <> label), do: {:b, label}
  defp parse_label("I-" <> label), do: {:i, label}
  defp parse_label(label), do: {:i, label}
end
