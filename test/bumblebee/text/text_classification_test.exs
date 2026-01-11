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

  test "scores sentence pairs correctly for cross-encoder reranking" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "cross-encoder/ms-marco-MiniLM-L-6-v2"})

    serving =
      Bumblebee.Text.TextClassification.text_classification(model_info, tokenizer,
        scores_function: :none
      )

    query = "How many people live in Berlin?"

    # Relevant document should score higher than irrelevant
    %{predictions: [%{score: relevant_score}]} =
      Nx.Serving.run(
        serving,
        {query, "Berlin has a population of 3,520,031 registered inhabitants."}
      )

    %{predictions: [%{score: irrelevant_score}]} =
      Nx.Serving.run(serving, {query, "New York City is famous for its skyscrapers."})

    assert relevant_score > irrelevant_score

    # Verify scores match Python sentence-transformers reference values
    assert_in_delta relevant_score, 8.76, 0.01
    assert_in_delta irrelevant_score, -11.24, 0.01
  end
end
