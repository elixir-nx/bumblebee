defmodule Bumblebee.Text.Generation.GrammarConstraint do
  @moduledoc false

  alias Bumblebee.Text.Generation.TokenTrie
  alias Bumblebee.Text.Generation.Stack
  alias EBNF.ParseState

  alias __MODULE__

  # Models a constraint

  defstruct [
    :token_trie,
    :grammar_encoding,
    :tokenizer,
    :start_rule_id,
    :start_rule_position,
    :rule_positions
  ]

  def create(grammar, root, tokenizer) do
    %ParseState{symbol_ids: symbols, grammar_encoding: encoding} = EBNF.encode(grammar)
    trie = TokenTrie.create(tokenizer)
    start_rule_id = Map.fetch!(symbols, root)
    rule_positions = get_rule_positions(encoding)

    %GrammarConstraint{
      token_trie: trie,
      grammar_encoding: encoding,
      tokenizer: tokenizer,
      start_rule_id: start_rule_id,
      start_rule_position: Map.fetch!(rule_positions, start_rule_id),
      rule_positions: rule_positions
    }
  end

  def init_stacks(constraint) do
    # stack will never exceed the grammar encoding size
    stack =
      Stack.new(length(constraint.grammar_encoding))
      |> Stack.push(constraint.start_rule_pos + 2)
      |> advance_stack()
  end

  defn advance_stack(stack) do
    if Nx.equal(Stack.length(stack), 0) do
      stack
    else
      top = Stack.peek(stack)

      if Nx.equal(top, 2) do
        stack
      else

        
      end
    end
  end

  defp get_rule_positions(grammar_encoding) do
    recur_get_rule_positions(grammar_encoding, 0, %{})
  end

  defp recur_get_rule_positions([0xFFFF], _pos, rule_positions), do: rule_positions

  defp recur_get_rule_positions([rule_id | rest], pos, rule_positions) do
    rule_positions = Map.put(rule_positions, rule_id, pos)

    case find_next_rule(rest, pos + 1) do
      {[_ | leftover], pos} ->
        recur_get_rule_positions(leftover, pos + 1, rule_positions)

      {[], _} ->
        rule_positions
    end
  end

  defp find_next_rule([0 | rest], pos) do
    {rest, pos + 1}
  end

  defp find_next_rule([rule_size | _] = leftover, pos) do
    leftover = Enum.drop(leftover, rule_size + 1)
    pos = pos + rule_size + 1

    case leftover do
      [0 | _] ->
        {leftover, pos}

      leftover ->
        find_next_rule(leftover, pos)
    end
  end
end
