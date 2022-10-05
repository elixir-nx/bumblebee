defmodule Bumblebee.Text.NER do
  @moduledoc """
  Utilities for extracting named-entities from token classification
  models.
  """
  alias Bumblebee.Utils.Tokenizers

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
    {aggregation_strategy, compiler_opts} = Keyword.pop(opts, :aggregation_strategy, nil)

    tensor_inputs =
      Bumblebee.apply_tokenizer(tokenizer, input,
        return_special_tokens_mask: true,
        return_offsets: true
      )

    {_init_fun, predict_fun} = Axon.build(model, compiler_opts)

    %{logits: logits} = predict_fun.(params, tensor_inputs)

    scores = Axon.Activations.softmax(logits)

    pre_entities = gather_pre_entities(tokenizer, input, tensor_inputs, scores)
    entities = aggregate(config, tokenizer, pre_entities, aggregation_strategy)

    Enum.filter(entities, fn
      %{"entity_group" => "O"} -> false
      _ -> true
    end)
  end

  @doc """
  Gathers metadata from inputs and scores for use in downstream
  entity aggregation task.
  """
  @spec gather_pre_entities(Bumblebee.Tokenizer.t(), string(), map(), Nx.t()) :: list()
  def gather_pre_entities(tokenizer, raw_input, tensor_inputs, scores) do
    {1, sequence_length, _} = Nx.shape(scores)
    flat_special_tokens_mask = Nx.to_flat_list(tensor_inputs["special_tokens_mask"])
    flat_input_ids = Nx.to_flat_list(tensor_inputs["input_ids"])
    flat_start_offsets = Nx.to_flat_list(tensor_inputs["start_offsets"])
    flat_end_offsets = Nx.to_flat_list(tensor_inputs["end_offsets"])

    # TODO: Optional offset mapping
    # TODO: Non-BPE tokenizers
    [
      0..(sequence_length - 1),
      flat_input_ids,
      flat_start_offsets,
      flat_end_offsets,
      flat_special_tokens_mask
    ]
    |> Enum.zip()
    |> Enum.filter(fn
      {_, _, _, _, 0} -> true
      {_, _, _, _, 1} -> false
    end)
    |> Enum.map(fn {token_index, input_id, start_ind, end_ind, _} ->
      word = Tokenizers.id_to_token(tokenizer.tokenizer, input_id)
      word_ref = String.slice(raw_input, start_ind, end_ind - start_ind)

      token_scores = scores[[0, token_index]]

      %{
        "word" => word,
        "scores" => token_scores,
        "start" => start_ind,
        "end" => end_ind,
        "index" => token_index,
        "is_subword" => String.length(word) != String.length(word_ref)
      }
    end)
  end

  @doc """
  Performs named-entity aggregation on the given pre-entities.
  """
  @spec aggregate(Bumblebee.ModelSpec.t(), Bumblebee.Tokenizer.t(), map(), nil | :simple) ::
          list()
  def aggregate(config, tokenizer, pre_entities, aggregation_strategy) do
    case aggregation_strategy do
      nil ->
        do_simple_aggregation(config, pre_entities)

      :simple ->
        config
        |> do_simple_aggregation(pre_entities)
        |> then(&group_entities(tokenizer, &1))
    end
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
