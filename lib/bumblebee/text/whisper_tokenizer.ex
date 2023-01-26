defmodule Bumblebee.Text.WhisperTokenizer do
  @moduledoc """
  Whisper tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      unk: "<|endoftext|>",
      bos: "<|endoftext|>",
      eos: "<|endoftext|>",
      pad: "<|endoftext|>"
    }
  )
end
