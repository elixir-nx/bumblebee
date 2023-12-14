defmodule Bumblebee.Text.BlenderbotTokenizer do
  @moduledoc """
  Blenderbot tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      unk: "<unk>",
      bos: "<s>",
      eos: "</s>",
      pad: "<pad>",
      sep: "</s>",
      cls: "<s>",
      mask: "<mask>"
    }
  )
end
