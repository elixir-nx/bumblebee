defmodule Bumblebee.Text.LlamaTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} =
               Bumblebee.load_tokenizer({:hf, "hf-internal-testing/llama-tokenizer"}, module: Bumblebee.Text.LlamaTokenizer)

      assert %Bumblebee.Text.LlamaTokenizer{} = tokenizer

      inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello everyobdy, how are you?"])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([[1, 15043, 1432, 711, 4518, 29892, 920, 526, 366, 29973]])
      )

      assert_equal(
        inputs["attention_mask"],
        Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      )
    end
  end
end
