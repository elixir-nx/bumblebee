defmodule Bumblebee.Text.Gliner.DecodeTest do
  use ExUnit.Case, async: true
  alias Bumblebee.Text.Gliner.Decode
  alias Bumblebee.Text.Gliner.Preprocessor

  test "synthetic punctuation is removed before overlap selection and deduplication" do
    result = decode("D", [{1.0, 0, 1}, {2.0, 0, 2}, {3.0, 1, 2}])
    assert [%{"text" => "D", "start" => 0, "end" => 1, "confidence" => score}] = result
    assert_in_delta score, 0.880797, 1.0e-6
  end

  test "empty and whitespace-only text never produce synthetic entities" do
    assert decode("", [{2.0, 0, 1}]) == []
    assert decode(" \t", [{2.0, 0, 1}]) == []
  end

  test "trailing whitespace is excluded from both surface and offsets" do
    assert [%{"text" => "A", "start" => 0, "end" => 1}] =
             decode("A \t", [{2.0, 0, 2}])
  end

  test "real terminal punctuation is preserved" do
    assert [%{"text" => "D.", "start" => 0, "end" => 2}] =
             decode("D.", [{2.0, 0, 2}])
  end

  test "projection retains Unicode codepoint offsets" do
    text = "👩🏽‍💻 Café"
    count = length(Preprocessor.words(text <> "."))

    assert [%{"text" => ^text, "start" => 0, "end" => ending}] =
             decode(text, [{2.0, 0, count}])

    assert ending == length(String.codepoints(text))
  end

  test "flat decoding maximizes total score rather than choosing the largest single span" do
    assert Decode.resolve_flat([{0.9, 0, 4}, {0.6, 0, 2}, {0.6, 2, 4}]) == [
             {0.6, 0, 2},
             {0.6, 2, 4}
           ]
  end

  test "ties prefer the larger compatible set then confidence/start/end rank" do
    assert Decode.resolve_flat([{1.0, 0, 4}, {0.5, 0, 2}, {0.5, 2, 4}]) == [
             {0.5, 0, 2},
             {0.5, 2, 4}
           ]

    assert Decode.resolve_flat([{0.5, 1, 3}, {0.5, 0, 2}, {0.4, 0, 2}]) == [{0.5, 0, 2}]
    assert Decode.resolve_flat([]) == []
  end

  test "threshold comparison remains inclusive without an epsilon" do
    assert [%{"confidence" => 0.5}] = decode("A", [{0.0, 0, 1}], 0.5)
    assert decode("A", [{0.0, 0, 1}], 0.50000001) == []
    assert [%{"confidence" => 0.5}] = decode("A", [{0.0, 0, 1}], 0.49999999)
  end

  defp decode(text, spans, threshold \\ 0.5) do
    augmented = if String.ends_with?(text, [".", "!", "?"]), do: text, else: text <> "."

    packed = %{
      text: augmented,
      original_text: text,
      original_length: length(String.codepoints(text)),
      words: Preprocessor.words(augmented),
      labels: ["person"]
    }

    output = %{
      "pair_logits" => Nx.tensor([Enum.map(spans, &elem(&1, 0))]),
      "null_logits" => Nx.tensor([[-10.0]]),
      "starts" => Nx.tensor([Enum.map(spans, &elem(&1, 1))]),
      "ends" => Nx.tensor([Enum.map(spans, &elem(&1, 2))]),
      "valid" => Nx.tensor([Enum.map(spans, fn _ -> 1 end)])
    }

    Decode.entities(packed, output, %{"pair_temperature" => 1.0, "abstention_threshold" => 0.5},
      threshold: threshold,
      include_spans: true,
      include_confidence: true
    )["entities"]["person"]
  end
end
