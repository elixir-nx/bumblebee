defmodule Bumblebee.Text.BertTokenizer do
  @moduledoc """
  BERT tokenizer.
  """

  defstruct [
    :tokenizer,
    special_tokens: %{unk: "[UNK]", sep: "[SEP]", pad: "[PAD]", cls: "[CLS]", mask: "[MASK]"}
  ]

  @behaviour Bumblebee.Tokenizer

  @impl true
  def apply(%{tokenizer: tokenizer, special_tokens: %{pad: pad_token}}, input, opts \\ []) do
    Bumblebee.Utils.Tokenizers.apply(tokenizer, input, pad_token, opts)
  end

  @impl true
  def decode(%{tokenizer: tokenizer}, ids) do
    Bumblebee.Utils.Tokenizers.decode(tokenizer, ids)
  end

  @impl true
  def id_to_token(%{tokenizer: tokenizer}, id) do
    Bumblebee.Utils.Tokenizers.id_to_token(tokenizer, id)
  end

  @impl true
  def token_to_id(%{tokenizer: tokenizer}, token) do
    Bumblebee.Utils.Tokenizers.token_to_id(tokenizer, token)
  end

  @impl true
  def special_tokens(tokenizer) do
    tokenizer.special_tokens
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(tokenizer, %{"tokenizer_file" => path, "special_tokens_map" => special_tokens_map}) do
      native_tokenizer = Bumblebee.Utils.Tokenizers.load!(path)

      special_tokens =
        Bumblebee.Shared.load_special_tokens(tokenizer.special_tokens, special_tokens_map)

      %{tokenizer | tokenizer: native_tokenizer, special_tokens: special_tokens}
    end
  end
end
