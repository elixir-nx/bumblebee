defmodule Bumblebee.Text.Gpt2TokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "encodes text" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})

    assert %Bumblebee.Text.Gpt2Tokenizer{} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Hello World"
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [15496, 2159]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1]
      ])
    )
  end
end
