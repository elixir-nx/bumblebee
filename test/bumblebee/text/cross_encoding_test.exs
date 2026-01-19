defmodule Bumblebee.Text.CrossEncodingTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  test "scores sentence pairs" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "cross-encoder/ms-marco-MiniLM-L-6-v2"})

    serving = Bumblebee.Text.cross_encoding(model_info, tokenizer)

    query = "How many people live in Berlin?"

    # Single pair
    assert %{score: score} =
             Nx.Serving.run(
               serving,
               {query, "Berlin has a population of 3,520,031 registered inhabitants."}
             )

    assert_in_delta score, 8.76, 0.01

    # Multiple pairs (batch)
    assert [%{score: relevant_score}, %{score: irrelevant_score}] =
             Nx.Serving.run(serving, [
               {query, "Berlin has a population of 3,520,031 registered inhabitants."},
               {query, "New York City is famous for its skyscrapers."}
             ])

    assert relevant_score > irrelevant_score
    assert_in_delta relevant_score, 8.76, 0.01
    assert_in_delta irrelevant_score, -11.24, 0.01
  end
end
