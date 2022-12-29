defmodule Bumblebee.Text.AlbertTokenizer do
  @moduledoc """
  ALBERT tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      bos: "[CLS]",
      eos: "[SEP]",
      unk: "<unk>",
      sep: "[SEP]",
      pad: "<pad>",
      cls: "[CLS]",
      mask: "[MASK]"
    }
  )
end
