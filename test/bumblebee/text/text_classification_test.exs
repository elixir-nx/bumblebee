defmodule Bumblebee.Text.TextClassificationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  test "returns top scored labels" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "cardiffnlp/twitter-roberta-base-emotion"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "FacebookAI/roberta-base"})

    serving = Bumblebee.Text.TextClassification.text_classification(model_info, tokenizer)

    text = "Cats are cute."

    assert %{
             predictions: [
               %{label: "optimism", score: _},
               %{label: "sadness", score: _},
               %{label: "anger", score: _},
               %{label: "joy", score: _}
             ]
           } = Nx.Serving.run(serving, text)
  end
end
