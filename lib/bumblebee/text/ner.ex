defmodule Bumblebee.Text.NER do
  @moduledoc """
  Utilities for extracting named-entities from token classification
  models.
  """
  alias Bumblebee.Utils.Tokenizers
  import Bumblebee.Utils.Nx

  @ignore_label "O"

  @doc """
  Performs end-to-end named entity recognition.

  This convenience function implements the end-to-end NER
  task, but offers less control over the extraction process.

  For more control, see the other functions in this module.

  ## Options

    * `:aggregation_strategy` - How to aggregate adjacent tokens.
      Token classification models output probabilities for each possible
      token class. The aggregation strategy takes scores for each token which
      possibly represents subwords and aggregates them back into words which
      are readily interpretable as entities. Supported aggregation strategies
      are `nil` and `:simple`. `nil` corresponds to no aggregation, which will
      simply return the max label for each token in the raw input. `:simple`
      corresponds to simple aggregation, which will group adjacent tokens
      of the same entity group as belonging to the same entity. Defaults to
      `nil`

    * `:length` - If provided, fixes length of provided inputs by truncating or
      padding to the given length. Otherwise inputs are padded to the maximum length
      input. Defaults to `nil`
  """
  @spec extract(
          Axon.t(),
          map(),
          Bumblebee.ModelSpec.t(),
          Bumblebee.Tokenizer.t(),
          String.t() | list(String.t()),
          keyword()
        ) :: list()
  def extract(model, params, spec, tokenizer, input, opts \\ [])

  def extract(model, params, spec, tokenizer, input, opts)
      when spec.architecture == :for_token_classification do
    {aggregation_strategy, opts} = Keyword.pop(opts, :aggregation_strategy, nil)
    {length, compiler_opts} = Keyword.pop(opts, :length, nil)

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, input,
        length: length,
        return_special_tokens_mask: true,
        return_offsets: true
      )

    {_init_fun, predict_fun} = Axon.build(model, compiler_opts)

    %{logits: logits} = predict_fun.(params, inputs)

    scores = Axon.Activations.softmax(logits)

    extract_from_scores(spec, tokenizer, input, inputs, scores,
      aggregation_strategy: aggregation_strategy
    )
  end

  def extract(_model, _params, %{architecture: arch}, _tokenizer, _input, _opts) do
    raise ArgumentError, "model spec must be a token classification model, got #{inspect(arch)}"
  end

  @doc """
  Extracts named entities from pre-computed scores and input
  tokens.

  ## Options

    * `:aggregation_strategy` - How to aggregate adjacent tokens.
      Token classification models output probabilities for each possible
      token class. The aggregation strategy takes scores for each token which
      possibly represents subwords and aggregates them back into words which
      are readily interpretable as entities. Supported aggregation strategies
      are `nil` and `:simple`. `nil` corresponds to no aggregation, which will
      simply return the max label for each token in the raw input. `:simple`
      corresponds to simple aggregation, which will group adjacent tokens
      of the same entity group as belonging to the same entity. Defaults to
      `nil`
  """
  @spec extract_from_scores(
          Bumblebee.ModelSpec.t(),
          Bumblebee.Tokenizer.t(),
          String.t(),
          map(),
          Nx.t()
        ) :: list()
  def extract_from_scores(spec, tokenizer, raw_input, inputs, scores, opts \\ []) do
    aggregation_strategy = opts[:aggregation_strategy]

    raw_input = List.wrap(raw_input)

    for {raw, tensors, score} <-
          Enum.zip([raw_input, to_batched(inputs, 1), to_batched(scores, 1)]) do
      tokenizer
      |> gather_pre_entities(raw, tensors, score)
      |> then(&aggregate(spec, tokenizer, &1, aggregation_strategy: aggregation_strategy))
      |> filter_entities(@ignore_label)
    end
  end

  defp gather_pre_entities(tokenizer, raw_input, inputs, scores) do
    {1, sequence_length, _} = Nx.shape(scores)
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

    for {token_idx, input_id, start_idx, end_idx, _special? = 0} <- token_infos do
      word = Bumblebee.Tokenizer.id_to_token(tokenizer, input_id)
      word_ref = String.slice(raw_input, start_idx, end_idx - start_idx)

      token_scores = scores[[0, token_idx]]

      %{
        "word" => word,
        "token_id" => input_id,
        "scores" => token_scores,
        "start" => start_idx,
        "end" => end_idx,
        "index" => token_idx,
        "is_subword" => String.length(word) != String.length(word_ref)
      }
    end
  end

  defp aggregate(spec, tokenizer, pre_entities, opts) do
    aggregation_strategy = opts[:aggregation_strategy]

    case aggregation_strategy do
      nil ->
        add_token_labels(spec, pre_entities)

      :simple ->
        spec
        |> add_token_labels(pre_entities)
        |> then(&group_entities(tokenizer, &1))
    end
  end

  defp filter_entities(entities, label) do
    Enum.filter(entities, fn %{"entity_group" => group} -> label != group end)
  end

  defp add_token_labels(spec, pre_entities) do
    Enum.map(pre_entities, fn pre_entity ->
      {scores, pre_entity} = Map.pop!(pre_entity, "scores")
      entity_idx = Nx.argmax(scores) |> Nx.to_number()
      score = scores[[entity_idx]] |> Nx.to_number()

      pre_entity
      |> Map.put("entity", spec.id_to_label[entity_idx])
      |> Map.put("score", score)
    end)
  end

  defp group_entities(tokenizer, entities) do
    {groups, _} =
      Enum.reduce(entities, {[], nil}, fn
        entity, {[], nil} ->
          {_bi, tag} = get_tag(entity["entity"])
          current_group = [entity]
          {[current_group], tag}

        entity, {[current_group | groups], last_tag} ->
          {bi, tag} = get_tag(entity["entity"])

          if tag == last_tag and bi != "B" do
            current_group = [entity | current_group]
            {[current_group | groups], tag}
          else
            new_current_group = [entity]
            {[new_current_group, current_group | groups], tag}
          end
      end)

    groups
    |> Enum.map(&group_sub_entities(tokenizer, &1))
    |> Enum.reverse()
  end

  defp group_sub_entities(tokenizer, [last_entity | _] = rev_group) do
    [first_entity | _] = group = Enum.reverse(rev_group)
    {_, tag} = get_tag(first_entity["entity"])
    scores = group |> Enum.map(fn %{"score" => score} -> score end) |> Nx.stack()

    tokens = Enum.map(group, fn %{"token_id" => id} -> id end)

    %{
      "entity_group" => tag,
      "score" => Nx.mean(scores),
      "word" => Tokenizers.decode(tokenizer.tokenizer, tokens),
      "start" => first_entity["start"],
      "end" => last_entity["end"]
    }
  end

  # Parse the BIO tagging format
  defp get_tag("B-" <> tag), do: {:b, tag}
  defp get_tag("I-" <> tag), do: {:i, tag}
  defp get_tag("O"), do: {:i, "O"}

  defp get_tag(label),
    do: raise(ArgumentError, "expected a label in the BIO format, got: #{inspect(label)}")
end
