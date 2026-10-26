defmodule Bumblebee.Text.Gliner.Preprocessor do
  @moduledoc false
  alias Tokenizers.{Encoding, Tokenizer}

  # Python's Unicode IGNORECASE includes these four non-ASCII letters in [a-z].
  @word_pattern ~r"(?:https?://[^\s]+|www\.[^\s]+)|[a-zİıſK0-9._%+-]+@[a-zİıſK0-9.-]+\.[a-zİıſK]{2,}|@[a-zİıſK0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S"iu

  def words(text) do
    Regex.scan(@word_pattern, text, return: :index)
    |> Enum.map(fn [{offset, size}] ->
      word = binary_part(text, offset, size)
      start = text |> binary_part(0, offset) |> String.codepoints() |> length()

      %{
        text: String.downcase(word, :greek),
        start: start,
        end: start + length(String.codepoints(word))
      }
    end)
  end

  def pack(text, labels, tokenizer) when is_binary(text) and is_list(labels) do
    original_text = text
    text = if String.ends_with?(text, [".", "!", "?"]), do: text, else: text <> "."
    words = words(text)
    schema = ["(", "[P]", "entities", "("] ++ Enum.flat_map(labels, &["[E]", &1]) ++ [")", ")"]
    tokens = schema ++ ["[SEP_TEXT]"] ++ Enum.map(words, & &1.text)

    encodings =
      Enum.map(tokens, fn token ->
        {:ok, encoding} = Tokenizer.encode(tokenizer, token, add_special_tokens: false)
        Encoding.get_ids(encoding)
      end)

    {positions, _} =
      Enum.map_reduce(encodings, 0, fn ids, position -> {position, position + length(ids)} end)

    ids = List.flatten(encodings)

    query_indices =
      labels
      |> Enum.with_index()
      |> Enum.map(fn {_, index} -> Enum.at(positions, 4 + index * 2) end)

    text_indices = Enum.drop(positions, length(schema) + 1)

    %{
      inputs: %{
        "input_ids" => Nx.tensor([ids], type: :s64),
        "attention_mask" => Nx.tensor([List.duplicate(1, length(ids))], type: :s64)
      },
      text_indices: Nx.tensor([text_indices], type: :s64),
      query_indices: Nx.tensor([query_indices], type: :s64),
      words: words,
      labels: labels,
      text: text,
      original_text: original_text,
      original_length: length(String.codepoints(original_text))
    }
  end
end
