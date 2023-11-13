defmodule Bumblebee.Text.XlmRobertaTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "xlm-roberta-base"})

      assert %Bumblebee.Text.XlmRobertaTokenizer{} = tokenizer

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, [
          "Test sentence with <mask>.",
          {"Question?", "Answer"}
        ])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([[0, 8647, 149_357, 678, 250_001, 6, 5, 2], [0, 68_185, 32, 2, 2, 130_373, 2, 1]])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0]])
      )

      assert_equal(
        inputs["token_type_ids"],
        Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
      )
    end
  end
end
