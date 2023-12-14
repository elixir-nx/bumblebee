defmodule Bumblebee.Text.GptNeoXTokenizer do
  @moduledoc """
  GPT-NeoX tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      unk: "<|endoftext|>",
      bos: "<|endoftext|>",
      eos: "<|endoftext|>",
      # GPT-NeoX doesn't originally have a pad token, however when necessary
      # we pad with the EOS token
      pad: "<|endoftext|>"
    }
  )
end
