defmodule Bumblebee.Text.MbartTokenizer do
  @moduledoc """
  mBART tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      eos: "</s>",
      unk: "<unk>",
      sep: "</s>",
      pad: "<pad>",
      cls: "<s>",
      mask: "<mask>"
    }
  )
end
