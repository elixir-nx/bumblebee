defmodule Bumblebee.Text.TokenClassificationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "correctly classifies labels with 1 sequence" do
      {:ok, model} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})
      labels = ["cooking", "traveling", "dancing"]

      zero_shot_serving = Bumblebee.Text.zero_shot_classification(model, tokenizer, labels)

      assert results = Nx.Serving.run(zero_shot_serving, "one day I will see the world")

      assert %{label: "traveling", score: _score} =
               Enum.max_by(results, fn %{score: score} -> score end)

      # assert_all_close(score, 0.9210067987442017)
    end

    test "correctly classifies labels with 2 sequences" do
      {:ok, model} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})
      labels = ["cooking", "traveling", "dancing"]

      zero_shot_serving = Bumblebee.Text.zero_shot_classification(model, tokenizer, labels)

      assert [results1, results2] =
               Nx.Serving.run(zero_shot_serving, [
                 "one day I will see the world",
                 "one day I will learn to salsa"
               ])

      assert %{label: "traveling", score: _score1} =
               Enum.max_by(results1, fn %{score: score} -> score end)

      assert %{label: "dancing", score: _score2} =
               Enum.max_by(results2, fn %{score: score} -> score end)

      # assert_all_close(score1, 0.9210067987442017)
      # TODO: This one is off by a pretty large factor? 0.82 vs 0.86
      # assert_all_close(score2, 0.8600915670394897)
    end

    test "correctly classifies batch with compilation set to true" do
      {:ok, model} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})
      labels = ["cooking", "traveling", "dancing"]

      zero_shot_serving =
        Bumblebee.Text.zero_shot_classification(model, tokenizer, labels,
          compile: [batch_size: 2, sequence_length: 32],
          defn_options: [compiler: EXLA]
        )

      assert [results1, results2] =
               Nx.Serving.run(zero_shot_serving, [
                 "one day I will see the world",
                 "one day I will learn to salsa"
               ])

      assert %{label: "traveling", score: _score1} =
               Enum.max_by(results1, fn %{score: score} -> score end)

      assert %{label: "dancing", score: _score2} =
               Enum.max_by(results2, fn %{score: score} -> score end)

      # assert_all_close(score1, 0.9210067987442017)
      # TODO: This one is off by a pretty large factor? 0.82 vs 0.86
      # assert_all_close(score2, 0.8600915670394897)
    end
  end
end
