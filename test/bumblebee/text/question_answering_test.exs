defmodule Bumblebee.Text.QuestionAnsweringTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  test "returns the most probable answer" do
    {:ok, roberta} = Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "FacebookAI/roberta-base"})

    serving = Bumblebee.Text.question_answering(roberta, tokenizer)

    input = %{question: "What's my name?", context: "My name is Sarah and I live in London."}

    assert %{
             results: [
               %{
                 text: "Sarah",
                 start: 11,
                 end: 16,
                 score: score
               }
             ]
           } = Nx.Serving.run(serving, input)

    assert_all_close(score, 0.8105)
  end

  test "supports multiple inputs" do
    {:ok, roberta} = Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "FacebookAI/roberta-base"})

    serving = Bumblebee.Text.question_answering(roberta, tokenizer)

    inputs = [
      %{question: "What's my name?", context: "My name is Sarah and I live in London."},
      %{question: "Where do I live?", context: "My name is Clara and I live in Berkeley."}
    ]

    assert [
             %{results: [%{text: "Sarah", start: 11, end: 16, score: _}]},
             %{results: [%{text: "Berkeley", start: 31, end: 39, score: _}]}
           ] = Nx.Serving.run(serving, inputs)
  end
end
