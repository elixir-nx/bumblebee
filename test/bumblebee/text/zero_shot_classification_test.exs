defmodule Bumblebee.Text.ZeroShotClassificationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "correctly classifies labels with 1 sequence" do
      {:ok, model} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})
      labels = ["cooking", "traveling", "dancing"]

      zero_shot_serving = Bumblebee.Text.zero_shot_classification(model, tokenizer, labels)

      output = Nx.Serving.run(zero_shot_serving, "one day I will see the world")

      assert %{
               predictions: [
                 %{label: "traveling", score: _},
                 %{label: "dancing", score: _},
                 %{label: "cooking", score: _}
               ]
             } = output

      assert %{label: "traveling", score: score} = Enum.max_by(output.predictions, & &1.score)
      assert_all_close(score, 0.9874)
    end

    test "correctly classifies labels with 2 sequences" do
      {:ok, model} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})
      labels = ["cooking", "traveling", "dancing"]

      zero_shot_serving = Bumblebee.Text.zero_shot_classification(model, tokenizer, labels)

      assert [output1, output2] =
               Nx.Serving.run(zero_shot_serving, [
                 "one day I will see the world",
                 "one day I will learn to salsa"
               ])

      assert %{label: "traveling", score: score1} = Enum.max_by(output1.predictions, & &1.score)
      assert_all_close(score1, 0.9874)

      assert %{label: "dancing", score: score2} = Enum.max_by(output2.predictions, & &1.score)
      assert_all_close(score2, 0.9585)
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

      assert [output1, output2] =
               Nx.Serving.run(zero_shot_serving, [
                 "one day I will see the world",
                 "one day I will learn to salsa"
               ])

      assert %{label: "traveling", score: score1} = Enum.max_by(output1.predictions, & &1.score)
      assert_all_close(score1, 0.9874)

      assert %{label: "dancing", score: score2} = Enum.max_by(output2.predictions, & &1.score)
      assert_all_close(score2, 0.9585)
    end
  end
end
