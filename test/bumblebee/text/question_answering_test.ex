defmodule Bumblebee.Text.QuestionAnsweringTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "returns top scored labels" do
      {:ok, roberta} = Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "roberta-base"})

      serving =
        Bumblebee.Text.question_answering(roberta, tokenizer,
          compile: [batch_size: 1, sequence_length: 32],
                    defn_options: [compiler: EXLA]

        )

      text_and_context = %{
        question: "What is your dads name ",
        context: "My  dads name is blue"
      }

      assert %{
               predictions: [
                 %{
                   text: "blue",
                   start: answer_start_index,
                   end: answer_end_index,
                   score: score
                 }
                 | rest
               ]
             } = Nx.Serving.run(serving, text_and_context)
    end
  end
end
