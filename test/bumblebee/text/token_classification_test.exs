defmodule Bumblebee.Text.TokenClassificationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "correctly extracts entities with :same aggregation" do
      assert {:ok, model_info} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      serving =
        Bumblebee.Text.TokenClassification.token_classification(model_info, tokenizer,
          aggregation: :same
        )

      text = "I went with Jane Doe to Atlanta and we talked to John Smith about Microsoft"

      assert %{entities: [jane, atlanta, john, microsoft]} = Nx.Serving.run(serving, text)

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

      # Offsets should be expressed in terms of bytes (note that é is 2 bytes)

      text = "Jane é John"

      assert %{
               entities: [%{start: 0, end: 4}, %{start: 8, end: 12}]
             } = Nx.Serving.run(serving, text)
    end

    for aggregation <- [:first, :max, :average] do
      test "correctly extracts entities with :#{aggregation} aggregation" do
        assert {:ok, model_info} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
        assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

        serving =
          Bumblebee.Text.TokenClassification.token_classification(model_info, tokenizer,
            aggregation: unquote(aggregation)
          )

        text = "I went with Jane Doe to Atlanta and we talked to John Smith about Microsoft"

        assert %{entities: [jane, atlanta, john, microsoft]} = Nx.Serving.run(serving, text)

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
    end

    test "correctly extracts entities with simple aggregation on batched input" do
      assert {:ok, model_info} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      serving =
        Bumblebee.Text.TokenClassification.token_classification(model_info, tokenizer,
          aggregation: :same
        )

      texts = [
        "I went with Jane Doe to Atlanta and we talked to John Smith about Microsoft",
        "John went to Philadelphia"
      ]

      assert [_first, %{entities: [john, philadelphia]}] = Nx.Serving.run(serving, texts)

      assert %{label: "PER", phrase: "John"} = john
      assert %{label: "LOC", phrase: "Philadelphia"} = philadelphia
    end
  end
end
