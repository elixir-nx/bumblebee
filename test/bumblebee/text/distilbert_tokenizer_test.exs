defmodule Bumblebee.Text.DistilbertTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "encodes text" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "distilbert-base-uncased"})

    assert %Bumblebee.Text.DistilbertTokenizer{} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [101, 3231, 6251, 2007, 103, 1012, 102],
        [101, 3160, 1029, 102, 3437, 102, 0]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0]
      ])
    )
  end

  test "with special tokens mask" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "distilbert-base-cased"})

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, ["Test sentence with [MASK]."],
        return_special_tokens_mask: true
      )

    assert_equal(inputs["special_tokens_mask"], Nx.tensor([[1, 0, 0, 0, 0, 0, 1]]))
  end

  test "with offsets" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "distilbert-base-cased"})

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, ["Test sentence with [MASK]."], return_offsets: true)

    assert_equal(inputs["start_offsets"], Nx.tensor([[0, 0, 5, 14, 19, 25, 0]]))
    assert_equal(inputs["end_offsets"], Nx.tensor([[0, 4, 13, 18, 25, 26, 0]]))
  end
end
