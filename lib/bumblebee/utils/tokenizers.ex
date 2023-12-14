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

    tokenizer =
      if length = opts[:length] do
        upper_bound_length = length |> List.wrap() |> Enum.max()

        Tokenizer.set_truncation(tokenizer,
          max_length: upper_bound_length,
          direction: opts[:truncate_direction]
        )
      else
        tokenizer
      end

    {:ok, encodings} =
      Tokenizer.encode_batch(tokenizer, input, add_special_tokens: opts[:add_special_tokens])

    length = opts[:length]

    pad_length =
      if is_number(length) do
        length
      else
        max_length =
          encodings
          |> Enum.map(&Encoding.n_tokens/1)
          |> Enum.max()

        case length do
          nil -> max_length
          lengths when is_list(lengths) -> find_bounding_length(max_length, lengths)
        end
      end

    pad_id = Tokenizer.token_to_id(tokenizer, pad_token)

    encodings =
      Enum.map(encodings, fn encoding ->
        Encoding.pad(encoding, pad_length,
          pad_id: pad_id,
          pad_token: pad_token,
          direction: opts[:pad_direction]
        )
      end)

    input_ids = encodings |> Enum.map(&Encoding.get_u32_ids/1) |> u32_binaries_to_tensor()

    encoded = %{"input_ids" => input_ids}

    encoded
    |> maybe_put_attention_mask(encodings, opts[:return_attention_mask])
    |> maybe_put_token_type_ids(encodings, opts[:return_token_type_ids])
    |> maybe_put_return_special_tokens_mask(encodings, opts[:return_special_tokens_mask])
    |> maybe_put_offsets(encodings, opts[:return_offsets])
  end

  defp find_bounding_length(max_length, lengths) do
    find_bounding_length(max_length, lengths, :infinity, 0)
  end

  defp find_bounding_length(max_length, [length | rest], bound, max) when length >= max_length do
    find_bounding_length(max_length, rest, min(bound, length), max(length, max))
  end

  defp find_bounding_length(max_length, [length | rest], bound, max) do
    find_bounding_length(max_length, rest, bound, max(length, max))
  end

  defp find_bounding_length(_max_length, [], bound, max), do: min(bound, max)

  defp maybe_put_attention_mask(encoded, encodings, return_attention_mask) do
    if return_attention_mask do
      attention_mask =
        encodings
        |> Enum.map(&Encoding.get_u32_attention_mask/1)
        |> u32_binaries_to_tensor()

      Map.put(encoded, "attention_mask", attention_mask)
    else
      encoded
    end
  end

  defp maybe_put_token_type_ids(encoded, encodings, return_token_type_ids) do
    if return_token_type_ids do
      token_type_ids =
        encodings
        |> Enum.map(&Encoding.get_u32_type_ids/1)
        |> u32_binaries_to_tensor()

      Map.put(encoded, "token_type_ids", token_type_ids)
    else
      encoded
    end
  end

  defp maybe_put_return_special_tokens_mask(encoded, encodings, return_special_tokens_mask) do
    if return_special_tokens_mask do
      special_tokens_mask =
        encodings
        |> Enum.map(&Encoding.get_u32_special_tokens_mask/1)
        |> u32_binaries_to_tensor()

      Map.put(encoded, "special_tokens_mask", special_tokens_mask)
    else
      encoded
    end
  end

  defp maybe_put_offsets(encoded, encodings, return_offsets) do
    if return_offsets do
      {batch_start_offsets, batch_end_offsets} =
        encodings
        |> Enum.map(fn seq ->
          seq |> Encoding.get_offsets() |> Enum.unzip()
        end)
        |> Enum.unzip()

      encoded
      |> Map.put("start_offsets", Nx.tensor(batch_start_offsets))
      |> Map.put("end_offsets", Nx.tensor(batch_end_offsets))
    else
      encoded
    end
  end

  defp u32_binaries_to_tensor(list) do
    list
    |> IO.iodata_to_binary()
    |> Nx.from_binary(:u32)
    |> Nx.reshape({length(list), :auto})
  end

  def decode(tokenizer, [ids | _] = batch_ids) when is_list(ids) do
    case Tokenizer.decode_batch(tokenizer, batch_ids) do
      {:ok, decoded} -> decoded
      {:error, term} -> raise "decoding failed with error: #{inspect(term)}"
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
    case Tokenizer.from_file(path) do
      {:ok, tokenizer} ->
        tokenizer
        |> Tokenizer.disable_padding()
        |> Tokenizer.disable_truncation()

      {:error, error} ->
        raise "failed to read tokenizer from file, reason: #{error}"
    end
  end
end
