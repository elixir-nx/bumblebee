defmodule Bumblebee.Text.TokenClassification do
  @moduledoc """
  High-level functions implementing token classification.
  """

  alias Bumblebee.Utils

  @type entity :: %{
          start: non_neg_integer(),
          end: non_neg_integer(),
          score: float(),
          label: String.t(),
          phrase: String.t()
        }

  @doc """
  Performs end-to-end token classification.

  This function can be used for tasks such as named entity recognition
  (NER) or part of speech tagging (POS).

  The recognized entities can optionally be aggregated into groups
  based on the given strategy.

  ## Options

    * `:aggregation` - an optional strategy for aggregating adjacent
      tokens. Token classification models output probabilities for
      each possible token class. The aggregation strategy takes scores
      for each token (which possibly represents subwords) and groups
      tokens into phrases which are readily interpretable as entities
      of a certain class. Supported aggregation strategies:

        * `nil` (default) - corresponds to no aggregation and returns
          the most likely label for each input token

        * `:same` - groups adjacent tokens with the same label. If
          the labels use beginning-inside-outside (BIO) tagging, the
          boundaries are respected and the prefix is omitted in the
          output labels

    * `:ignored_labels` - the labels to ignore in the final output.
      The labels should be specified without BIO prefix. Defaults to
      `["O"]`

    * `:length` - the length to pad/truncate the prompts to. Fixing
      the length to a certain value allows for caching model compilation
      across different prompts. By default prompts are padded to
      match the longest one

    * `:defn_options` - the options for to JIT compilation. Defaults
      to `[]`

  """
  @spec extract(
          Axon.t(),
          map(),
          Bumblebee.ModelSpec.t(),
          Bumblebee.Tokenizer.t(),
          String.t() | list(String.t()),
          keyword()
        ) :: list(list(entity()))
  def extract(model, params, spec, tokenizer, text, opts \\ [])

  def extract(model, params, spec, tokenizer, text, opts)
      when spec.architecture == :for_token_classification do
    opts =
      Keyword.validate!(opts, [:aggregation, :length, ignored_labels: ["O"], defn_options: []])

    aggregation = opts[:aggregation]
    ignored_labels = opts[:ignored_labels]
    length = opts[:length]
    defn_options = opts[:defn_options]

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, text,
        length: length,
        return_special_tokens_mask: true,
        return_offsets: true
      )

    {_init_fun, predict_fun} = Axon.build(model, defn_options)

    %{logits: logits} = predict_fun.(params, inputs)
    scores = Axon.Activations.softmax(logits)

    [List.wrap(text), Utils.Nx.to_batched(inputs, 1), Utils.Nx.to_batched(scores, 1)]
    |> Enum.zip_with(fn [text, inputs, scores] ->
      scores
      |> gather_raw_entities(tokenizer, text, inputs)
      |> aggregate(spec, tokenizer, aggregation)
      |> filter_entities(ignored_labels)
    end)
  end

  def extract(_model, _params, %{architecture: arch}, _tokenizer, _text, _opts) do
    raise ArgumentError, "expected a model for token classification, got #{inspect(arch)}"
  end

  defp gather_raw_entities(scores, tokenizer, text, inputs) do
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

    for {token_idx, token_id, start_idx, end_idx, _special? = 0} <- token_infos do
      token = Bumblebee.Tokenizer.id_to_token(tokenizer, token_id)
      token_ref = String.slice(text, start_idx, end_idx - start_idx)

      token_scores = scores[[0, token_idx]]

      %{
        token: token,
        token_id: token_id,
        scores: token_scores,
        start: start_idx,
        end: end_idx,
        index: token_idx,
        is_subword: String.length(token) != String.length(token_ref)
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
