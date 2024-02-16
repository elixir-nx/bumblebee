defmodule Bumblebee.Text.FillMaskTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  test "returns top scored tokens" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})

    serving = Bumblebee.Text.FillMask.fill_mask(model_info, tokenizer)

    text = "The capital of [MASK] is Paris."

    assert %{
             predictions: [
               %{score: _, token: "france"},
               %{score: _, token: "brittany"},
               %{score: _, token: "algeria"},
               %{score: _, token: "department"},
               %{score: _, token: "reunion"}
             ]
           } = Nx.Serving.run(serving, text)
  end

  test "raises when there isn't exactly one mask token" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})

    serving = Bumblebee.Text.FillMask.fill_mask(model_info, tokenizer)

    assert_raise ArgumentError,
                 ~s/expected exactly one occurrence of [MASK], got: 0 in "The capital of France is Paris."/,
                 fn ->
                   Nx.Serving.run(serving, "The capital of France is Paris.")
                 end

    assert_raise ArgumentError,
                 ~s/expected exactly one occurrence of [MASK], got: 2 in "The [MASK] of [MASK] is Paris."/,
                 fn ->
                   Nx.Serving.run(serving, "The [MASK] of [MASK] is Paris.")
                 end
  end
end
