defmodule Bumblebee.Text.CamembertTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "encodes text" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "camembert-base"})

    assert %Bumblebee.Text.CamembertTokenizer{} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with <mask>."
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [5, 9115, 22625, 1466, 32004, 21, 9, 6]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1]
      ])
    )
  end
end
