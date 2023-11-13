defmodule Bumblebee.Text.BartTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-base"})

      assert %Bumblebee.Text.BartTokenizer{} = tokenizer

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, [
          "Test sentence with [MASK].",
          {"Question?", "Answer"}
        ])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([
          [0, 34_603, 3645, 19, 646, 32_804, 530, 8174, 2],
          [0, 45_641, 116, 2, 2, 33_683, 2, 1, 1]
        ])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 0, 0]
        ])
      )

      assert_equal(
        inputs["token_type_ids"],
        Nx.tensor([
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
      )
    end
  end
end
