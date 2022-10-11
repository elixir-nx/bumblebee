defmodule Bumblebee.Text.BartTokenizer do
  @doc """
  BART tokenizer.
  """

  defstruct [:tokenizer]

  @behaviour Bumblebee.Tokenizer

  @impl true
  def apply(%{tokenizer: tokenizer}, input, opts \\ []) do
    Bumblebee.Utils.Tokenizers.apply(tokenizer, input, "<pad>", opts)
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
    %{
      bos: "<s>",
      eos: "</s>",
      unk: "<unk>",
      sep: "</s>",
      pad: "<pad>",
      cls: "<s>",
      mask: "<mask>"
    }
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(tokenizer, %{"tokenizer_file" => path}) do
      native_tokenizer = Bumblebee.Utils.Tokenizers.load!(path)
      %{tokenizer | tokenizer: native_tokenizer}
    end
  end
end
