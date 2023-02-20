defmodule Bumblebee.Text.T5TokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "t5-small"})

      assert %Bumblebee.Text.T5Tokenizer{} = tokenizer

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, ["translate English to German: How old are you?"])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([[13959, 1566, 12, 2968, 10, 571, 625, 33, 25, 58, 1]])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      )
    end
  end
end
