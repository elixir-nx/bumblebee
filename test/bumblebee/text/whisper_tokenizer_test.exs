defmodule Bumblebee.Text.WhisperTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})

      assert %Bumblebee.Text.WhisperTokenizer{} = tokenizer

      inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello world"])

      assert_equal(inputs["input_ids"], Nx.tensor([[50_258, 50_363, 15_947, 1002, 50_257]]))
      assert_equal(inputs["attention_mask"], Nx.tensor([[1, 1, 1, 1, 1]]))
    end
  end
end
