defmodule Bumblebee.Text.NERTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "correctly extracts entities with simple aggregation" do
      assert {:ok, model, params, spec} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      input = "I went with Jane Doe to Atlanta and we talked to John Smith about Microsoft"

      assert [[jane, atlanta, john, microsoft]] =
               Bumblebee.Text.NER.extract(model, params, spec, tokenizer, input,
                 aggregation_strategy: :simple,
                 compiler: EXLA
               )

      assert %{
               "entity_group" => "PER",
               "score" => _jane_score,
               "word" => "Jane Doe",
               "start" => 12,
               "end" => 20
             } = jane

      assert %{
               "entity_group" => "LOC",
               "score" => _atlanta_score,
               "word" => "Atlanta",
               "start" => 24,
               "end" => 31
             } = atlanta

      assert %{
               "entity_group" => "PER",
               "score" => _john_score,
               "word" => "John Smith",
               "start" => 49,
               "end" => 59
             } = john

      assert %{
               "entity_group" => "ORG",
               "score" => _microsoft_score,
               "word" => "Microsoft",
               "start" => 66,
               "end" => 75
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
               Bumblebee.Text.NER.extract(model, params, spec, tokenizer, inputs,
                 aggregation_strategy: :simple,
                 compiler: EXLA
               )

      assert %{"entity_group" => "PER", "word" => "John"} = john
      assert %{"entity_group" => "LOC", "word" => "Philadelphia"} = philadelphia
    end
  end
end
