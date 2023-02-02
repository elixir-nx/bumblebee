defmodule Bumblebee.Text.QuestionAnsweringTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "returns top scored labels" do
      {:ok, roberta} = Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "roberta-base"})
      serving = Bumblebee.Text.QuestionAnswering.answer_question(roberta, tokenizer)

      text_and_context = %{
        question: "What is your favirote color",
        context: "My favriote color is blue"
      }

      assert %{
               predictions: [
                 %{
                   text: answers,
                   start: answer_start_index,
                   end: answer_end_index
                 }
                 | rest
               ]
             } = Nx.Serving.run(serving, text_and_context) |> IO.inspect
    end
  end
end
