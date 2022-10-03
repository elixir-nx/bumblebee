defmodule Bumblebee.Text.AlbertTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "albert-base-v2"})

      assert %Bumblebee.Text.AlbertTokenizer{} = tokenizer

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, [
          "Test sentence with [MASK].",
          {"Question?", "Answer"}
        ])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([
          [2, 1289, 5123, 29, 4, 13, 9, 3],
          [2, 1301, 60, 3, 1623, 3, 0, 0]
        ])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 0, 0]
        ])
      )

      assert_equal(
        inputs["token_type_ids"],
        Nx.tensor([
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 0, 0]
        ])
      )
    end

    test "encoding model input pads and truncates to length" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "albert-base-v2"})

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, ["foo", "foo bar", "foo bar baz bang buzz"],
          length: 8
        )

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([
          [2, 4310, 111, 3, 0, 0, 0, 0],
          [2, 4310, 111, 748, 3, 0, 0, 0],
          [2, 4310, 111, 748, 19674, 6582, 9122, 3]
        ])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
      )
    end
  end
end
