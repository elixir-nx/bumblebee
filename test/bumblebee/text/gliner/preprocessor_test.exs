defmodule Bumblebee.Text.Gliner.PreprocessorTest do
  use ExUnit.Case, async: true
  alias Bumblebee.Text.Gliner.Preprocessor

  test "Python IGNORECASE keeps dotted-I emails and handles intact" do
    assert Preprocessor.words("İNFO@example.org @ıpek") == [
             %{text: "i̇nfo@example.org", start: 0, end: 16},
             %{text: "@ıpek", start: 17, end: 22}
           ]
  end

  test "Python lower uses contextual final sigma and preserves original offsets" do
    assert Preprocessor.words("ΟΣ ΟΣΑ ΣΟΣ ΑΣ; İ") == [
             %{text: "ος", start: 0, end: 2},
             %{text: "οσα", start: 3, end: 6},
             %{text: "σος", start: 7, end: 10},
             %{text: "ας", start: 11, end: 13},
             %{text: ";", start: 13, end: 14},
             %{text: "i̇", start: 15, end: 16}
           ]
  end
end
