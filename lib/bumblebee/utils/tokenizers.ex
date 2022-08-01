defmodule Bumblebee.Utils.Tokenizers do
  @moduledoc false

  # Shared logic for implementing Bumblebee.Tokenizer using the
  # tokenizers package.

  alias Tokenizers.{Tokenizer, Encoding}

  def apply(tokenizer, input, add_special_tokens, pad_token, pad_direction) do
    input = List.wrap(input)

    {:ok, encodings} = Tokenizer.encode(tokenizer, input, add_special_tokens: add_special_tokens)

    max_length =
      encodings
      |> Enum.map(&Encoding.n_tokens/1)
      |> Enum.max()

    pad_id = Tokenizer.token_to_id(tokenizer, pad_token)

    encodings =
      Enum.map(
        encodings,
        &Encoding.pad(&1, max_length,
          pad_id: pad_id,
          pad_token: pad_token,
          direction: pad_direction
        )
      )

    input_ids = Enum.map(encodings, &Encoding.get_ids/1)
    attention_mask = Enum.map(encodings, &Encoding.get_attention_mask/1)
    token_type_ids = Enum.map(encodings, &Encoding.get_type_ids/1)

    %{
      "input_ids" => Nx.tensor(input_ids),
      "attention_mask" => Nx.tensor(attention_mask),
      "token_type_ids" => Nx.tensor(token_type_ids)
    }
  end

  def decode(tokenizer, ids) do
    Tokenizer.decode(tokenizer, ids)
  end

  def id_to_token(tokenizer, id) do
    Tokenizer.id_to_token(tokenizer, id)
  end

  def token_to_id(tokenizer, token) do
    Tokenizer.token_to_id(tokenizer, token)
  end

  def load!(path) do
    case Tokenizers.Tokenizer.from_file(path) do
      {:ok, tokenizer} -> tokenizer
      {:error, error} -> raise "failed to read tokenizer from file, reason: #{error}"
    end
  end
end
