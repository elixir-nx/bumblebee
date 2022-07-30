defmodule Bumblebee.Text.Gpt2TokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})

      assert %Bumblebee.Text.Gpt2Tokenizer{} = tokenizer

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, [
          "Hello World"
        ])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([
          [32, 922, 6827, 351, 220, 50256]
        ])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([
          [1, 1, 1, 1, 1, 1]
        ])
      )
    end
  end
end
