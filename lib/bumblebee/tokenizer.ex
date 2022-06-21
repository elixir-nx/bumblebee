defmodule Bumblebee.Tokenizer do
  @moduledoc """
  An interface for configuring and applying tokenizers.

  A tokenizer is used to convert raw text data into model input.

  Every module implementing this behaviour is expected to also define
  a configuration struct.
  """

  @typedoc """
  Tokenizer configuration and metadata.
  """
  @type t :: %{
          optional(atom()) => term(),
          __struct__: atom()
        }

  @type input :: String.t() | {String.t(), String.t()}

  @doc """
  Performs tokenization and encoding on the given input.
  """
  @callback apply(t(), input() | list(input), add_special_tokens :: boolean()) :: any()
end
