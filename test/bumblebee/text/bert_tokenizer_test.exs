defmodule Bumblebee.Text.BertTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

      assert %Bumblebee.Text.BertTokenizer{} = tokenizer

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

      assert_equal(
        inputs["token_type_ids"],
        Nx.tensor([
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 0]
        ])
      )
    end

    test "encoding with special tokens mask" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      inputs =
        Bumblebee.apply_tokenizer(
          tokenizer,
          [
            "Test sentence with [MASK]."
          ],
          return_special_tokens_mask: true
        )

      assert_equal(inputs["special_tokens_mask"], Nx.tensor([[1, 0, 0, 0, 0, 0, 1]]))
    end

    test "encoding with offsets" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      inputs =
        Bumblebee.apply_tokenizer(
          tokenizer,
          [
            "Test sentence with [MASK]."
          ],
          return_offsets: true
        )

      assert_equal(inputs["start_offsets"], Nx.tensor([[0, 0, 5, 14, 19, 25, 0]]))
      assert_equal(inputs["end_offsets"], Nx.tensor([[0, 4, 13, 18, 25, 26, 0]]))
    end

    test "encoding with multiple lengths" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      inputs = Bumblebee.apply_tokenizer(tokenizer, "This is short.", length: [8, 16])

      assert {1, 8} = Nx.shape(inputs["input_ids"])

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, "This is definitely much longer than the above.",
          length: [8, 16]
        )

      assert {1, 16} = Nx.shape(inputs["input_ids"])
    end
  end
end
