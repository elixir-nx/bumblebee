defmodule Bumblebee.Text.PreTrainedTokenizer do
  alias Bumblebee.Shared

  options = [
    add_special_tokens: [
      default: true,
      doc: "whether to add special tokens during tokenization"
    ],
    length: [
      default: nil,
      doc: """
      applies fixed length padding or truncation to the given input if set. Can be either
      a specific number or a list of numbers. When a list is given, the smallest number
      that exceeds all input lengths is used as the padding length
      """
    ],
    pad_direction: [
      default: :right,
      doc: "the padding direction, either `:right` or `:left`"
    ],
    truncate_direction: [
      default: :right,
      doc: "the truncation direction, either `:right` or `:left`"
    ],
    return_attention_mask: [
      default: true,
      doc: """
      whether to return attention mask for encoded sequence. The mask is a boolean tensor
      indicating which tokens are padding and should effectively be ignored by the model
      """
    ],
    return_token_type_ids: [
      default: true,
      doc: "whether to return token type ids for encoded sequence"
    ],
    return_special_tokens_mask: [
      default: false,
      doc: """
      whether to return special tokens mask for encoded sequence. The mask is a boolean
      tensor indicating which tokens are special
      """
    ],
    return_offsets: [
      default: false,
      doc: """
      whether to return token offsets for encoded sequence. This tensor includes a list of
      position pairs that map tokens to the input text
      """
    ],
    return_length: [
      default: false,
      doc: """
      whether to return the sequence length. The length is the effective number of tokens,
      so it is calculated after truncation, but does not include padding
      """
    ],
    template_options: [
      default: [],
      doc: """
      options configuring the tokenization template, specific to the given tokenizer type.
      Recognised options are:

        * `:language_token` - for tokenizers: `:nllb`

      """
    ]
  ]

  @moduledoc """
  Wraps a pre-trained tokenizer from the `Tokenizers` library.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [
              :native_tokenizer,
              :type,
              special_tokens: %{},
              additional_special_tokens: []
            ] ++ Shared.option_defaults(options)

  alias Tokenizers.{Tokenizer, Encoding}

  @behaviour Bumblebee.Tokenizer
  @behaviour Bumblebee.Configurable

  @tokenizer_types %{
    albert: %{
      special_tokens: %{
        bos: "[CLS]",
        eos: "[SEP]",
        unk: "<unk>",
        sep: "[SEP]",
        pad: "<pad>",
        cls: "[CLS]",
        mask: "[MASK]"
      }
    },
    bart: %{
      special_tokens: %{
        bos: "<s>",
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      }
    },
    bert: %{
      special_tokens: %{unk: "[UNK]", sep: "[SEP]", pad: "[PAD]", cls: "[CLS]", mask: "[MASK]"}
    },
    blenderbot: %{
      special_tokens: %{
        unk: "<unk>",
        bos: "<s>",
        eos: "</s>",
        pad: "<pad>",
        sep: "</s>",
        cls: "<s>",
        mask: "<mask>"
      }
    },
    camembert: %{
      special_tokens: %{
        bos: "<s>",
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      }
    },
    clip: %{
      special_tokens: %{unk: "<|endoftext|>", pad: "<|endoftext|>", eos: "<|endoftext|>"}
    },
    code_gen: %{
      special_tokens: %{
        unk: "<|endoftext|>",
        bos: "<|endoftext|>",
        eos: "<|endoftext|>"
      }
    },
    distilbert: %{
      special_tokens: %{unk: "[UNK]", sep: "[SEP]", pad: "[PAD]", cls: "[CLS]", mask: "[MASK]"}
    },
    gemma: %{
      special_tokens: %{
        unk: "<unk>",
        bos: "<bos>",
        eos: "<eos>",
        pad: "<pad>"
      }
    },
    gpt_neo_x: %{
      special_tokens: %{
        unk: "<|endoftext|>",
        bos: "<|endoftext|>",
        eos: "<|endoftext|>"
      }
    },
    gpt2: %{
      special_tokens: %{
        unk: "<|endoftext|>",
        bos: "<|endoftext|>",
        eos: "<|endoftext|>"
      }
    },
    layout_lm: %{
      special_tokens: %{unk: "[UNK]", sep: "[SEP]", pad: "[PAD]", cls: "[CLS]", mask: "[MASK]"}
    },
    llama: %{
      special_tokens: %{
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>"
      }
    },
    mbart: %{
      special_tokens: %{
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      }
    },
    mpnet: %{
      special_tokens: %{
        bos: "<s>",
        eos: "</s>",
        unk: "[UNK]",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      }
    },
    nllb: %{
      special_tokens: %{
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      },
      default_template_options: [language_token: "eng_Latn"]
    },
    qwen2: %{
      special_tokens: %{
        unk: "<|endoftext|>",
        eos: "<|endoftext|>",
        pad: "<|endoftext|>"
      }
    },
    roberta: %{
      special_tokens: %{
        bos: "<s>",
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      }
    },
    smollm3: %{
      special_tokens: %{
        eos: "<|im_end|>",
        pad: "<|im_end|>"
      }
    },
    t5: %{
      special_tokens: %{
        bos: "<s>",
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      }
    },
    whisper: %{
      special_tokens: %{
        unk: "<|endoftext|>",
        bos: "<|endoftext|>",
        eos: "<|endoftext|>",
        pad: "<|endoftext|>"
      }
    },
    xlm_roberta: %{
      special_tokens: %{
        bos: "<s>",
        eos: "</s>",
        unk: "<unk>",
        sep: "</s>",
        pad: "<pad>",
        cls: "<s>",
        mask: "<mask>"
      }
    }
  }

  @impl true
  def config(%{native_tokenizer: nil}, _opts) do
    raise ArgumentError,
          "configuring #{inspect(__MODULE__)} from scratch is not supported," <>
            " you need to load an existing tokenizer first"
  end

  def config(tokenizer, opts) do
    tokenizer = Shared.put_config_attrs(tokenizer, opts)

    # Doing truncation manually after tokenization could truncate
    # special tokens added by a template post-processor. By setting
    # truncation upfront, the tokenizer will apply it before the
    # post-processor accounting for the extra special tokens
    tokenizer =
      if Keyword.has_key?(opts, :length) or Keyword.has_key?(opts, :truncation_direction) do
        update_truncation(tokenizer)
      else
        tokenizer
      end

    if Keyword.has_key?(opts, :template_options) do
      set_template(tokenizer)
    else
      tokenizer
    end
  end

  defp update_truncation(%{length: nil} = tokenizer) do
    update_in(tokenizer.native_tokenizer, &Tokenizer.disable_truncation/1)
  end

  defp update_truncation(%{length: length} = tokenizer) do
    upper_bound_length = length |> List.wrap() |> Enum.max()

    update_in(
      tokenizer.native_tokenizer,
      &Tokenizer.set_truncation(&1,
        max_length: upper_bound_length,
        direction: tokenizer.truncate_direction
      )
    )
  end

  defp set_template(%{type: :nllb} = tokenizer) do
    language_token = Keyword.fetch!(tokenizer.template_options, :language_token)
    eos_token = tokenizer.special_tokens.eos

    set_template_postprocessor(
      tokenizer,
      "#{language_token} $A #{eos_token}",
      "#{language_token} $A $B #{eos_token}",
      [language_token, eos_token]
    )
  end

  defp set_template(%{type: type} = tokenizer) do
    if tokenizer.template_options != [] do
      raise ArgumentError,
            "#{inspect(type)} tokenizer expects no :template_options," <>
              " got: #{inspect(tokenizer.template_options)}"
    end

    tokenizer
  end

  defp set_template_postprocessor(tokenizer, single, pair, special_tokens) do
    post_processor =
      Tokenizers.PostProcessor.template(
        single: single,
        pair: pair,
        special_tokens:
          for token <- special_tokens do
            {token, Tokenizer.token_to_id(tokenizer.native_tokenizer, token)}
          end
      )

    update_in(tokenizer.native_tokenizer, &Tokenizer.set_post_processor(&1, post_processor))
  end

  @impl true
  def apply(tokenizer, input) do
    input = List.wrap(input)

    # Some tokenizers don't specify a PAD token, in which case we use
    # the EOS token for padding by default
    pad_token =
      tokenizer.special_tokens[:pad] ||
        tokenizer.special_tokens[:eos] ||
        raise ArgumentError,
              "expected the tokenizer to defined a padding token, but none was found"

    {:ok, encodings} =
      Tokenizer.encode_batch(tokenizer.native_tokenizer, input,
        add_special_tokens: tokenizer.add_special_tokens
      )

    lengths = Enum.map(encodings, &Encoding.n_tokens/1)

    pad_length =
      if is_number(tokenizer.length) do
        tokenizer.length
      else
        max_length = Enum.max(lengths)

        case tokenizer.length do
          nil -> max_length
          lengths when is_list(lengths) -> find_bounding_length(max_length, lengths)
        end
      end

    pad_id = Tokenizer.token_to_id(tokenizer.native_tokenizer, pad_token)

    encodings =
      Enum.map(encodings, fn encoding ->
        Encoding.pad(encoding, pad_length,
          pad_id: pad_id,
          pad_token: pad_token,
          direction: tokenizer.pad_direction
        )
      end)

    input_ids = encodings |> Enum.map(&Encoding.get_u32_ids/1) |> u32_binaries_to_tensor()

    encoded = %{"input_ids" => input_ids}

    encoded
    |> maybe_put_attention_mask(encodings, tokenizer.return_attention_mask)
    |> maybe_put_token_type_ids(encodings, tokenizer.return_token_type_ids)
    |> maybe_put_return_special_tokens_mask(encodings, tokenizer.return_special_tokens_mask)
    |> maybe_put_offsets(encodings, tokenizer.return_offsets)
    |> maybe_put_lengths(lengths, tokenizer.return_length)
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

  defp maybe_put_lengths(encoded, lengths, return_length) do
    if return_length do
      Map.put(encoded, "length", Nx.tensor(lengths))
    else
      encoded
    end
  end

  defp u32_binaries_to_tensor(list) do
    binary = IO.iodata_to_binary(list)

    if binary == <<>> do
      raise ArgumentError,
            "the tokenizer returned zero tokens. Depending on the tokenizer," <>
              " this may happen for blank input. You should check if the input is blank" <>
              " before attempting tokenization"
    end

    binary
    |> Nx.from_binary(:u32)
    |> Nx.reshape({length(list), :auto})
  end

  @impl true
  def decode(tokenizer, [ids | _] = batch_ids) when is_list(ids) do
    case Tokenizer.decode_batch(tokenizer.native_tokenizer, batch_ids) do
      {:ok, decoded} -> decoded
      {:error, term} -> raise "decoding failed with error: #{inspect(term)}"
    end
  end

  def decode(tokenizer, ids) do
    case Tokenizer.decode(tokenizer.native_tokenizer, ids) do
      {:ok, decoded} -> decoded
      {:error, term} -> raise "decoding failed with error: #{inspect(term)}"
    end
  end

  @impl true
  def id_to_token(tokenizer, id) do
    Tokenizer.id_to_token(tokenizer.native_tokenizer, id)
  end

  @impl true
  def token_to_id(tokenizer, token) do
    Tokenizer.token_to_id(tokenizer.native_tokenizer, token)
  end

  @impl true
  def special_tokens(tokenizer) do
    tokenizer.special_tokens
  end

  @impl true
  def additional_special_tokens(tokenizer) do
    tokenizer.additional_special_tokens
  end

  @doc false
  def tokenizer_types(), do: @tokenizer_types

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(tokenizer, %{
          "tokenizer_file" => path,
          "special_tokens_map" => special_tokens_map
        }) do
      native_tokenizer =
        case Tokenizer.from_file(path) do
          {:ok, tokenizer} ->
            tokenizer
            |> Tokenizer.disable_padding()
            |> Tokenizer.disable_truncation()

          {:error, error} ->
            raise "failed to read tokenizer from file, reason: #{error}"
        end

      tokenizer_types = Bumblebee.Text.PreTrainedTokenizer.tokenizer_types()

      unless Map.has_key?(tokenizer_types, tokenizer.type) do
        types = tokenizer_types |> Map.keys() |> Enum.sort()

        raise ArgumentError,
              "expected tokenizer type to be one of: #{Enum.map_join(types, ", ", &inspect/1)}," <>
                " but got: #{inspect(tokenizer.type)}"
      end

      tokenizer_type = %{special_tokens: special_tokens} = tokenizer_types[tokenizer.type]

      special_tokens = load_special_tokens(special_tokens, special_tokens_map)

      additional_special_tokens =
        case special_tokens_map do
          %{"additional_special_tokens" => tokens} ->
            for token <- tokens, do: load_token(token), into: MapSet.new()

          _ ->
            []
        end

      template_options = tokenizer_type[:default_template_options] || []

      %{
        tokenizer
        | native_tokenizer: native_tokenizer,
          special_tokens: special_tokens,
          additional_special_tokens: additional_special_tokens
      }
      |> @for.config(template_options: template_options)
    end

    defp load_special_tokens(special_tokens, data) do
      for {key, default_token} <- special_tokens, into: %{} do
        token =
          if token = data["#{key}_token"] do
            load_token(token)
          else
            default_token
          end

        {key, token}
      end
    end

    defp load_token(token) when is_binary(token), do: token
    defp load_token(%{"content" => token}) when is_binary(token), do: token
  end
end
