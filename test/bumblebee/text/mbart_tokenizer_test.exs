defmodule Bumblebee.Text.MbartTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/mbart-large-cc25"})

      assert %Bumblebee.Text.MbartTokenizer{} = tokenizer

      inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello, my dog is cute <mask>"])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([
          [35378, 4, 759, 10269, 83, 99942, 250_026, 2, 250_004]
        ])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([
          [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
      )
    end
  end
end
