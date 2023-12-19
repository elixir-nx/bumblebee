defmodule Bumblebee.Tokenizer do
  @moduledoc """
  An interface for configuring and applying tokenizers.

  A tokenizer is used to convert raw text data into model input.

  Every module implementing this behaviour is expected to also define
  a configuration struct.
  """

  @type t :: struct()

  @type input :: String.t() | {String.t(), String.t()}
  @type token :: String.t()
  @type token_id :: non_neg_integer()

  @typedoc """
  A type corresponding to a special token in the vocabulary.

  ## Common types

    * `:bos` - a token representing the beginning of a sentence

    * `:eos` - a token representing the end of a sentence

    * `:unk` - a token representing an out-of-vocabulary token

    * `:sep` - a token separating two different sentences in the same
      input

    * `:pad` - a token added when processing a batch of sequences with
      different length

    * `:cls` - a token representing the class of the input

    * `:mask` - a token representing a masked token, used for masked
      language modeling tasks

  """
  @type special_token_type :: atom()

  @doc """
  Performs tokenization and encoding on the given input.
  """
  @callback apply(t(), input() | list(input())) :: any()

  @doc """
  Decodes a list of token ids into a sentence.
  """
  @callback decode(t(), list(token_id()) | list(list(token_id()))) :: String.t()

  @doc """
  Converts the given token into the corresponding numeric id.
  """
  @callback token_to_id(t(), token()) :: token_id()

  @doc """
  Converts the given token id the corresponding token.
  """
  @callback id_to_token(t(), token_id()) :: token()

  @doc """
  Returns a map with special tokens.
  """
  @callback special_tokens(t()) :: %{special_token_type() => token()}

  @doc """
  Returns a list with extra special tokens, in addition to the named
  `special_tokens/1`.
  """
  @callback additional_special_tokens(t()) :: MapSet.t(token())

  @doc """
  Decodes a list of token ids into a sentence.
  """
  @spec decode(
          t(),
          token() | list(token_id()) | list(list(token_id())) | Nx.Tensor.t()
        ) :: String.t()
  def decode(%module{} = tokenizer, ids) do
    ids = with %Nx.Tensor{} <- ids, do: Nx.to_list(ids)
    ids = List.wrap(ids)
    module.decode(tokenizer, ids)
  end

  @doc """
  Converts the given token into the corresponding numeric id.
  """
  @spec token_to_id(t(), token()) :: token_id()
  def token_to_id(%module{} = tokenizer, token) do
    module.token_to_id(tokenizer, token)
  end

  @doc """
  Converts the given token id the corresponding token.
  """
  @spec token_to_id(t(), token_id()) :: token()
  def id_to_token(%module{} = tokenizer, id) do
    module.id_to_token(tokenizer, id)
  end

  @doc """
  Returns a special token by name.
  """
  @spec special_token(t(), special_token_type()) :: token() | nil
  def special_token(%module{} = tokenizer, type) do
    special_tokens = module.special_tokens(tokenizer)
    special_tokens[type]
  end

  @doc """
  Returns id of a special token by name.
  """
  @spec special_token_id(t(), special_token_type()) :: token_id() | nil
  def special_token_id(tokenizer, type) do
    if token = special_token(tokenizer, type) do
      token_to_id(tokenizer, token)
    end
  end

  @doc """
  Returns all special tokens, including any extra tokens.
  """
  @spec all_special_tokens(t()) :: list(token_id())
  def all_special_tokens(%module{} = tokenizer) do
    special_tokens = module.special_tokens(tokenizer)
    additional_special_tokens = module.additional_special_tokens(tokenizer)
    for {_type, token} <- special_tokens, do: token, into: additional_special_tokens
  end
end
