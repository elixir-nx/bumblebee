defmodule Bumblebee.Text.WhisperTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "encodes text" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})

    assert %Bumblebee.Text.WhisperTokenizer{} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello world"])

    assert_equal(inputs["input_ids"], Nx.tensor([[50258, 50363, 15947, 1002, 50257]]))
    assert_equal(inputs["attention_mask"], Nx.tensor([[1, 1, 1, 1, 1]]))
  end
end
