defmodule Bumblebee.Text.ClipTokenizer do
  @moduledoc """
  CLIP tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{unk: "<|endoftext|>", pad: "<|endoftext|>", eos: "<|endoftext|>"}
  )
end
