defmodule Bumblebee.Text.BertTokenizer do
  @doc """
  BERT tokenizer.
  """

  defstruct [:tokenizer]

  @behaviour Bumblebee.Tokenizer

  @impl true
  def apply(%{tokenizer: tokenizer}, input, add_special_tokens) do
    Bumblebee.Utils.Tokenizers.apply(tokenizer, input, add_special_tokens, "[PAD]")
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
  def special_tokens(_tokenizer) do
    %{unk: "[UNK]", sep: "[SEP]", pad: "[PAD]", cls: "[CLS]", mask: "[MASK]"}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, %{"tokenizer_file" => path}) do
      tokenizer = Bumblebee.Utils.Tokenizers.load!(path)
      %{config | tokenizer: tokenizer}
    end
  end
end
