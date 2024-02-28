defmodule Bumblebee.Text.Generation.TokenTrie do
  @moduledoc false

  # Internal data structure used in constrained sampling

  alias Bumblebee.Text.PreTrainedTokenizer
  alias __MODULE__

  defstruct [:tokens, :trie, :eos_token_id]

  @leaf -1

  @doc """
  Returns the token encoded by the given ID.
  """
  def id_to_token(%TokenTrie{tokens: tokens}, id) do
    Map.fetch!(tokens, id)
  end

  @doc """
  Returns the number of tokens in the trie.
  """
  def n_tokens(%TokenTrie{tokens: tokens}) do
    length(tokens)
  end

  @doc """
  Creates a trie from the vocabulary in the given tokenizer.
  """
  def create(%PreTrainedTokenizer{native_tokenizer: tokenizer, special_tokens: %{eos: eos_token}}) do
    vocab = Tokenizers.Tokenizer.get_vocab(tokenizer)
    eos_token_id = Map.fetch!(vocab, eos_token)

    tokens =
      Map.new(vocab, fn {token, id} ->
        # TODO: Special cases for GPT2 and Llama
        {id, String.to_charlist(token)}
      end)

    trie =
      Enum.reduce(tokens, %{}, fn {token_id, token_bytes}, acc ->
        insert_into_trie(acc, token_bytes, token_id)
      end)

    %TokenTrie{tokens: tokens, trie: trie, eos_token_id: eos_token_id}
  end

  ## Helpers

  defp insert_into_trie(trie, token_bytes, token_id) do
    do_insert_into_trie(trie, token_bytes, token_id)
  end

  defp do_insert_into_trie(trie, [], token_id), do: Map.put(trie, @leaf, token_id)

  defp do_insert_into_trie(trie, [byte | rest_bytes], token_id) do
    current = Map.get(trie, byte, %{})
    updated = do_insert_into_trie(current, rest_bytes, token_id)
    Map.put(trie, byte, updated)
  end
end
