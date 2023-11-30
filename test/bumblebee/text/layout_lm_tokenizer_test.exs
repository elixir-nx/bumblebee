defmodule Bumblebee.Text.LayoutLmTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "encodes text" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "microsoft/layoutlm-base-uncased"})

    assert %Bumblebee.Text.LayoutLmTokenizer{} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[101, 3231, 6251, 2007, 103, 1012, 102], [101, 3160, 1029, 102, 3437, 102, 0]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0]])
    )
  end
end
