defmodule Bumblebee.Text.NER do
  @moduledoc """
  Utilities for extracting named-entities from token classification
  models.
  """
  alias Bumblebee.Utils.Tokenizers

  @default_ignore_label "O"

  @doc """
  Performs end-to-end named entity recognition.

  This convenience function implements the end-to-end NER
  task, but offers less control over the extraction process.

  For more control, see the other functions in this module.

  ## Options

    * `:aggregation_strategy` - How to aggregate adjacent tokens.
  """
  @spec extract(
          Bumblebee.ModelSpec.t(),
          Bumblebee.Tokenizer.t(),
          Axon.t(),
          map(),
          map(),
          keyword()
        ) :: list()
  def extract(config, tokenizer, model, params, input, opts \\ []) do
    {aggregation_strategy, opts} = Keyword.pop(opts, :aggregation_strategy, nil)
    {ignore_label, compiler_opts} = Keyword.pop(opts, :ignore_label, @default_ignore_label)

    tensor_inputs =
      Bumblebee.apply_tokenizer(tokenizer, input,
        return_special_tokens_mask: true,
        return_offsets: true
      )

    {_init_fun, predict_fun} = Axon.build(model, compiler_opts)

    %{logits: logits} = predict_fun.(params, tensor_inputs)

    scores = Axon.Activations.softmax(logits)

    extract_from_scores(config, tokenizer, input, tensor_inputs, scores,
      aggregation_strategy: aggregation_strategy,
      ignore_label: ignore_label
    )
  end

  @doc """
  Extracts named entities from pre-computed scores.

  ## Options

    * `:aggregation_strategy` - How to aggregate adjacent tokens.
  """
  @spec extract_from_scores(
          Bumblebee.ModelSpec.t(),
          Bumblebee.Tokenizer.t(),
          String.t(),
          map(),
          Nx.t()
        ) :: list()
  def extract_from_scores(config, tokenizer, raw_input, tensor_inputs, scores, opts \\ []) do
    aggregation_strategy = opts[:aggregation_strategy]
    ignore_label = opts[:ignore_label] || @default_ignore_label

    tokenizer
    |> gather_pre_entities(raw_input, tensor_inputs, scores)
    |> then(&aggregate(config, tokenizer, &1, aggregation_strategy: aggregation_strategy))
    |> filter_entities(ignore_label)
  end

  defp gather_pre_entities(tokenizer, raw_input, tensor_inputs, scores) do
    {1, sequence_length, _} = Nx.shape(scores)
    flat_special_tokens_mask = Nx.to_flat_list(tensor_inputs["special_tokens_mask"])
    flat_input_ids = Nx.to_flat_list(tensor_inputs["input_ids"])
    flat_start_offsets = Nx.to_flat_list(tensor_inputs["start_offsets"])
    flat_end_offsets = Nx.to_flat_list(tensor_inputs["end_offsets"])

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

    for {token_idx, input_id, start_idx, end_idx, _special? = 0} <- token_infos do
      word = Bumblebee.Tokenizer.id_to_token(tokenizer, input_id)
      word_ref = String.slice(raw_input, start_idx, end_idx - start_idx)

      token_scores = scores[[0, token_idx]]

      %{
        "word" => word,
        "scores" => token_scores,
        "start" => start_idx,
        "end" => end_idx,
        "index" => token_idx,
        "is_subword" => String.length(word) != String.length(word_ref)
      }
    end
  end

  defp aggregate(config, tokenizer, pre_entities, opts) do
    aggregation_strategy = opts[:aggregation_strategy]

    case aggregation_strategy do
      nil ->
        do_simple_aggregation(config, pre_entities)

      :simple ->
        config
        |> do_simple_aggregation(pre_entities)
        |> then(&group_entities(tokenizer, &1))
    end
  end

  defp filter_entities(entities, label) do
    Enum.filter(entities, fn %{"entity_group" => group} -> label != group end)
  end

  defp do_simple_aggregation(config, pre_entities) do
    Enum.map(pre_entities, fn pre_entity ->
      {scores, pre_entity} = Map.pop!(pre_entity, "scores")
      entity_idx = Nx.argmax(scores) |> Nx.to_number()
      score = scores[[entity_idx]]

      pre_entity
      |> Map.put("entity", config.id2label[entity_idx])
      |> Map.put("score", score)
    end)
  end

  defp group_entities(tokenizer, entities) do
    {groups, current_group, _, _} =
      Enum.reduce(entities, {[], [], nil, nil}, fn entity,
                                                   {groups, current_group, last_bi, last_tag} ->
        {bi, tag} = get_tag(entity["entity"])

        cond do
          last_bi == nil and last_tag == nil ->
            current_group = [entity | current_group]
            {groups, current_group, bi, tag}

          tag == last_tag and bi != "B" ->
            current_group = [entity | current_group]
            {groups, current_group, bi, tag}

          true ->
            group = group_sub_entities(tokenizer, current_group)
            current_group = [entity]
            {[group | groups], current_group, bi, tag}
        end
      end)

    case current_group do
      [] ->
        Enum.reverse(groups)

      [_ | _] ->
        Enum.reverse([group_sub_entities(tokenizer, current_group) | groups])
    end
  end

  defp group_sub_entities(tokenizer, [last_entity | _] = rev_group) do
    [first_entity | _] = group = Enum.reverse(rev_group)
    {_, tag} = get_tag(first_entity["entity"])
    scores = group |> Enum.map(fn %{"score" => score} -> score end) |> Nx.stack()

    tokens =
      group
      |> Enum.map(fn %{"word" => word} -> Tokenizers.token_to_id(tokenizer.tokenizer, word) end)

    %{
      "entity_group" => tag,
      "score" => Nx.mean(scores),
      "word" => Tokenizers.decode(tokenizer.tokenizer, tokens),
      "start" => first_entity["start"],
      "end" => last_entity["end"]
    }
  end

  defp get_tag(<<"B-"::binary, tag::binary>>), do: {"B", tag}
  defp get_tag(<<"I-"::binary, tag::binary>>), do: {"I", tag}
  defp get_tag(<<"O">>), do: {"I", "O"}
  defp get_tag(_), do: raise(ArgumentError, "entity labels are invalid for NER task")
end
