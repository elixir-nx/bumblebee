defmodule Bumblebee.Utils.Tokenizers do
  @moduledoc false

  # Shared logic for implementing Bumblebee.Tokenizer using the
  # tokenizers package.

  alias Tokenizers.{Tokenizer, Encoding}

  def apply(tokenizer, input, pad_token, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        add_special_tokens: true,
        pad_direction: :right,
        truncate_direction: :right,
        length: nil,
        return_attention_mask: true,
        return_token_type_ids: true,
        return_special_tokens_mask: false,
        return_offsets: false
      )

    input = List.wrap(input)

    {:ok, encodings} =
      Tokenizer.encode(tokenizer, input, add_special_tokens: opts[:add_special_tokens])

    length =
      if length = opts[:length] do
        length
      else
        encodings
        |> Enum.map(&Encoding.n_tokens/1)
        |> Enum.max()
      end

    pad_id = Tokenizer.token_to_id(tokenizer, pad_token)

    encodings =
      Enum.map(encodings, fn seq ->
        seq
        |> Encoding.pad(length,
          pad_id: pad_id,
          pad_token: pad_token,
          direction: opts[:pad_direction]
        )
        |> Encoding.truncate(length, direction: opts[:truncate_direction])
      end)

    input_ids = Enum.map(encodings, &Encoding.get_ids/1)

    encoded = %{"input_ids" => Nx.tensor(input_ids)}

    encoded
    |> maybe_put_attention_mask(encodings, opts[:return_attention_mask])
    |> maybe_put_token_type_ids(encodings, opts[:return_token_type_ids])
    |> maybe_put_return_special_tokens_mask(encodings, opts[:return_special_tokens_mask])
    |> maybe_put_offsets(encodings, opts[:return_offsets])
  end

  defp maybe_put_attention_mask(encoded, encodings, return_attention_mask) do
    if return_attention_mask do
      attention_mask = Enum.map(encodings, &Encoding.get_attention_mask/1)
      Map.put(encoded, "attention_mask", Nx.tensor(attention_mask))
    else
      encoded
    end
  end

  defp maybe_put_token_type_ids(encoded, encodings, return_token_type_ids) do
    if return_token_type_ids do
      token_type_ids = Enum.map(encodings, &Encoding.get_type_ids/1)
      Map.put(encoded, "token_type_ids", Nx.tensor(token_type_ids))
    else
      encoded
    end
  end

  defp maybe_put_return_special_tokens_mask(encoded, encodings, return_special_tokens_mask) do
    if return_special_tokens_mask do
      special_tokens_mask = Enum.map(encodings, &Encoding.get_special_tokens_mask/1)
      Map.put(encoded, "special_tokens_mask", Nx.tensor(special_tokens_mask))
    else
      encoded
    end
  end

  defp maybe_put_offsets(encoded, encodings, return_offsets) do
    if return_offsets do
      {batch_start_offets, batch_end_offsets} =
        encodings
        |> Enum.map(&Encoding.get_offsets/1)
        |> Enum.unzip()

      encoded
      |> Map.put("start_offsets", Nx.tensor(batch_start_offets))
      |> Map.put("end_offsets", Nx.tensor(batch_end_offsets))
    else
      encoded
    end
  end

  def decode(tokenizer, ids) do
    case Tokenizer.decode(tokenizer, ids) do
      {:ok, decoded} -> decoded
      {:error, term} -> raise "decoding failed with error: #{inspect(term)}"
    end
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
