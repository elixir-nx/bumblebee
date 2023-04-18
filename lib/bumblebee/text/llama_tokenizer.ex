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
    }
  )
end
