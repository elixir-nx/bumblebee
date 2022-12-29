defmodule Bumblebee.Text.Gpt2Tokenizer do
  @moduledoc """
  GPT-2 tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      unk: "<|endoftext|>",
      bos: "<|endoftext|>",
      eos: "<|endoftext|>",
      # GPT-2 doesn't originally have a pad token, however when necessary
      # we pad with the EOS token
      pad: "<|endoftext|>"
    }
  )
end
