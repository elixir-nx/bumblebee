defmodule Bumblebee.Text.QuestionAnsweringTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "returns top scored labels" do
      {:ok, roberta} =
        Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"},
          architecture: :for_question_answering
        )

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "roberta-base"})

      serving =
        Bumblebee.Text.question_answering(roberta, tokenizer,
          compile: [batch_size: 1, sequence_length: 32],
          defn_options: [compiler: EXLA]
        )

      text_and_context = %{
        question: "What is my name",
        context: "My name is blackeuler"
      }

      assert %{
               results: [
                 %{
                   text: "blackeuler",
                   start: 11,
                   end: 21,
                   score: _score
                 }
               ]
             } = Nx.Serving.run(serving, text_and_context)
    end
  end
end
