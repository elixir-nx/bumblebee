defmodule Bumblebee.Text.T5Tokenizer do
  @moduledoc """
  T5 tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      bos: "<s>",
      eos: "</s>",
      unk: "<unk>",
      sep: "</s>",
      pad: "<pad>",
      cls: "<s>",
      mask: "<mask>"
    }
  )
end
