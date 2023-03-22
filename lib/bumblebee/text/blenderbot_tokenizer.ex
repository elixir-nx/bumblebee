defmodule Bumblebee.Text.BlenderbotTokenizer do
  @moduledoc """
  Blenderbot tokenizer.
  """

  @behaviour Bumblebee.Text.Conversation

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

  @impl true
  def conversation_history_to_text(_tokenizer, history) do
    history
    |> Enum.reverse()
    |> Enum.map_join("  ", fn
      # The model generates a leading space, for user inputs we add one
      {:user, text} -> " " <> text
      {:generated, text} -> text
    end)
  end
end
