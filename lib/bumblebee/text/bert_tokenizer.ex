defmodule Bumblebee.Text.BertTokenizer do
  @doc """
  BERT tokenizer.
  """

  defstruct [:tokenizer]

  alias Tokenizers.{Tokenizer, Encoding}

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  @behaviour Bumblebee.Tokenizer

  @impl true
  def apply(%{tokenizer: tokenizer}, sentences, add_special_tokens) do
    sentences = List.wrap(sentences)

    {:ok, encodings} =
      Tokenizer.encode(tokenizer, sentences, add_special_tokens: add_special_tokens)

    max_length =
      encodings
      |> Enum.map(&Encoding.n_tokens/1)
      |> Enum.max()

    encodings = Enum.map(encodings, &Encoding.pad(&1, max_length))

    input_ids = Enum.map(encodings, &Encoding.get_ids/1)
    attention_mask = Enum.map(encodings, &Encoding.get_attention_mask/1)
    token_type_ids = Enum.map(encodings, &Encoding.get_type_ids/1)

    %{
      "input_ids" => Nx.tensor(input_ids),
      "attention_mask" => Nx.tensor(attention_mask),
      "token_type_ids" => Nx.tensor(token_type_ids)
    }
  end

  @impl true
  def decode(%{tokenizer: tokenizer}, ids) do
    Tokenizer.decode(tokenizer, ids)
  end

  @impl true
  def id_to_token(%{tokenizer: tokenizer}, id) do
    Tokenizer.id_to_token(tokenizer, id)
  end

  @impl true
  def token_to_id(%{tokenizer: tokenizer}, token) do
    Tokenizer.token_to_id(tokenizer, token)
  end

  @impl true
  def special_tokens(_tokenizer) do
    %{unknown: "[UNK]", separator: "[SEP]", padding: "[PAD]", class: "[CLS]", mask: "[MASK]"}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    @compile {:no_warn_undefined, Tokenizers.Tokenizer}

    def load(config, %{"tokenizer_file" => path}) do
      tokenizer =
        case Tokenizers.Tokenizer.from_file(path) do
          {:ok, tokenizer} -> tokenizer
          {:error, error} -> raise "failed to read tokenizer from file, reason: #{error}"
        end

      %{config | tokenizer: tokenizer}
    end
  end
end
