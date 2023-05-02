defmodule Bumblebee.Text.LlamaTokenizer do
  @moduledoc """
  Llama tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      eos: "</s>",
      unk: "<unk>",
      sep: "</s>",
      # Llama doesn't originally have a pad token, however when necessary
      # we pad with the EOS token
      pad: "</s>"
    }
  )
end
