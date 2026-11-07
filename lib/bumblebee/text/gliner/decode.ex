defmodule Bumblebee.Text.Gliner.Decode do
  @moduledoc false

  def entities(packed, outputs, config, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.5)
    null = outputs["null_logits"] |> sigmoid() |> Nx.to_flat_list()

    probabilities =
      outputs["pair_logits"]
      |> Nx.divide(config["pair_temperature"])
      |> sigmoid()
      |> Nx.to_flat_list()

    starts = Nx.to_flat_list(outputs["starts"])
    endings = Nx.to_flat_list(outputs["ends"])
    valid = Nx.to_flat_list(outputs["valid"])
    scores = Enum.chunk_every(probabilities, length(starts))

    entities =
      Enum.zip([packed.labels, null, scores])
      |> Map.new(fn {label, abstention, scores} ->
        candidates = Enum.zip([scores, starts, endings, valid])
        spans = decode_query(candidates, abstention, config, threshold, packed, opts)

        {label, spans}
      end)

    %{"entities" => entities}
  end

  defp sigmoid(x) do
    Axon.Activations.sigmoid(x)
  end

  defp decode_query(candidates, abstention, config, threshold, packed, opts) do
    if abstention > config["abstention_threshold"] do
      []
    else
      candidates
      |> Enum.filter(fn {score, _, _, valid} -> valid == 1 and score >= threshold end)
      |> Enum.flat_map(&original_span(&1, packed))
      |> resolve_flat()
      |> Enum.map(&format_span(&1, packed, opts))
    end
  end

  def resolve_flat(spans) do
    ordered =
      spans
      |> Enum.sort_by(&rank/1)
      |> Enum.uniq_by(fn {_, start, ending} -> {start, ending} end)
      |> Enum.sort_by(fn {score, start, ending} -> {ending, start, -score} end)

    best =
      Enum.reduce(ordered, [{0.0, []}], fn {score, start, _} = span, best ->
        predecessor =
          Enum.reduce_while(ordered, 0, fn
            {_, _, ending}, count when ending <= start -> {:cont, count + 1}
            _, count -> {:halt, count}
          end)

        {previous_score, previous_spans} = Enum.at(best, predecessor)
        with_span = {previous_score + score, [span | previous_spans]}
        without_span = List.last(best)
        best ++ [choose(with_span, without_span)]
      end)

    best |> List.last() |> elem(1) |> Enum.sort_by(&rank/1)
  end

  defp choose({a, left}, {b, right}) do
    cond do
      a > b -> {a, left}
      a < b -> {b, right}
      length(left) > length(right) -> {a, left}
      length(left) < length(right) -> {b, right}
      Enum.sort(Enum.map(left, &rank/1)) < Enum.sort(Enum.map(right, &rank/1)) -> {a, left}
      true -> {b, right}
    end
  end

  defp rank({score, start, ending}), do: {-score, start, ending}

  # Project before overlap resolution: a synthetic-only span must not compete
  # with real entities, and spans differing only by the suffix are duplicates.
  defp original_span({score, start, ending, _valid}, packed) do
    first = Enum.at(packed.words, start).start
    last = min(Enum.at(packed.words, ending - 1).end, packed.original_length)
    raw = slice(packed.original_text, first, max(last - first, 0))
    left_trimmed = String.trim_leading(raw)
    text = String.trim_trailing(left_trimmed)

    if text == "" do
      []
    else
      first = first + length(String.codepoints(raw)) - length(String.codepoints(left_trimmed))
      [{score, first, first + length(String.codepoints(text))}]
    end
  end

  defp format_span({score, char_start, char_end}, packed, opts) do
    surface = slice(packed.original_text, char_start, char_end - char_start)

    fields = %{"text" => surface}

    fields =
      if opts[:include_spans],
        do: Map.merge(fields, %{"start" => char_start, "end" => char_end}),
        else: fields

    fields = if opts[:include_confidence], do: Map.put(fields, "confidence", score), else: fields
    if opts[:include_spans] || opts[:include_confidence], do: fields, else: surface
  end

  defp slice(text, start, length) do
    text |> String.codepoints() |> Enum.slice(start, length) |> Enum.join()
  end
end
