defmodule Bumblebee.Text.RobertaTokenizer do
  @moduledoc """
  RoBERTa tokenizer.
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
