defmodule Bumblebee.Text.TokenClassificationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "correctly extracts entities with :same aggregation" do
      assert {:ok, model, params, spec} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      text = "I went with Jane Doe to Atlanta and we talked to John Smith about Microsoft"

      assert [[jane, atlanta, john, microsoft]] =
               Bumblebee.Text.TokenClassification.extract(model, params, spec, tokenizer, text,
                 aggregation: :same
               )

      assert %{
               label: "PER",
               score: _jane_score,
               phrase: "Jane Doe",
               start: 12,
               end: 20
             } = jane

      assert %{
               label: "LOC",
               score: _atlanta_score,
               phrase: "Atlanta",
               start: 24,
               end: 31
             } = atlanta

      assert %{
               label: "PER",
               score: _john_score,
               phrase: "John Smith",
               start: 49,
               end: 59
             } = john

      assert %{
               label: "ORG",
               score: _microsoft_score,
               phrase: "Microsoft",
               start: 66,
               end: 75
             } = microsoft
    end

    test "correctly extracts entities with simple aggregation on batched input" do
      assert {:ok, model, params, spec} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      inputs = [
        "I went with Jane Doe to Atlanta and we talked to John Smith about Microsoft",
        "John went to Philadelphia"
      ]

      assert [_first, [john, philadelphia]] =
               Bumblebee.Text.TokenClassification.extract(model, params, spec, tokenizer, inputs,
                 aggregation: :same
               )

      assert %{label: "PER", phrase: "John"} = john
      assert %{label: "LOC", phrase: "Philadelphia"} = philadelphia
    end
  end
end
