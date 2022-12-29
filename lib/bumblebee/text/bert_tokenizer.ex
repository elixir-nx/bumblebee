defmodule Bumblebee.Text.BertTokenizer do
  @moduledoc """
  BERT tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{unk: "[UNK]", sep: "[SEP]", pad: "[PAD]", cls: "[CLS]", mask: "[MASK]"}
  )
end
