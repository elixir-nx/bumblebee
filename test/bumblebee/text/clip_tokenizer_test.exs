defmodule Bumblebee.Text.ClipTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-base-patch32"})

      assert %Bumblebee.Text.ClipTokenizer{} = tokenizer

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, [
          "a photo of a cat",
          "a photo of a dog"
        ])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([
          [49_406, 320, 1125, 539, 320, 2368, 49_407],
          [49_406, 320, 1125, 539, 320, 1929, 49_407]
        ])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1]
        ])
      )
    end
  end
end
