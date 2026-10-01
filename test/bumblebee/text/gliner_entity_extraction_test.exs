defmodule Bumblebee.Text.GlinerEntityExtractionTest do
  use ExUnit.Case, async: false
  alias Bumblebee.Text.GlinerEntityExtraction, as: Extraction
  @moduletag Bumblebee.TestHelpers.serving_test_tags()

  setup_all do
    reference =
      Path.expand("../../fixtures/gliner/ner_reference.json", __DIR__)
      |> File.read!()
      |> Jason.decode!()

    repository =
      case System.get_env("GLINER_MODEL_DIR") do
        nil -> {:hf, reference["model"], revision: reference["revision"]}
        path -> {:local, path}
      end

    {:ok, model} =
      Bumblebee.Text.load_entity_extraction(repository, defn_options: [compiler: EXLA])

    %{model: model, examples: reference["examples"]}
  end

  test "entity decisions and confidence match Python on representative inputs", %{
    model: model,
    examples: examples
  } do
    for example <- examples do
      actual =
        Extraction.extract_entities(model, example["text"], example["labels"],
          include_spans: true,
          include_confidence: true
        )

      assert Map.keys(actual["entities"]) |> Enum.sort() ==
               Map.keys(example["prediction"]["entities"]) |> Enum.sort()

      for {label, expected} <- example["prediction"]["entities"] do
        assert length(actual["entities"][label]) == length(expected)

        for {actual, expected} <- Enum.zip(actual["entities"][label], expected) do
          assert Map.delete(actual, "confidence") == Map.delete(expected, "confidence")
          assert_in_delta actual["confidence"], expected["confidence"], 1.0e-5
        end
      end
    end
  end

  test "serves concurrent documents, deduplicates labels and handles empty labels", %{
    model: model
  } do
    serving = Bumblebee.Text.entity_extraction(model, include_spans: true)
    start_supervised!({Nx.Serving, serving: serving, name: __MODULE__})

    input = %{
      text: "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
      labels: ["company", "person", "company"]
    }

    expected = %{
      "entities" => %{
        "company" => [%{"text" => "Apple", "start" => 0, "end" => 5}],
        "person" => [%{"text" => "Tim Cook", "start" => 10, "end" => 18}]
      }
    }

    for result <-
          Task.async_stream(1..4, fn _ -> Nx.Serving.batched_run(__MODULE__, input) end,
            timeout: 120_000
          ) do
      assert {:ok, ^expected} = result
    end

    assert Nx.Serving.batched_run(__MODULE__, %{text: "Ignored", labels: []}) == %{}

    assert Extraction.extract_entities(model, "Ignored", []) == %{}
  end

  test "validates inputs and limits before model execution", %{model: model} do
    assert_raise ArgumentError, ~r/nonempty strings/, fn ->
      Extraction.extract_entities(model, "Hello", [""])
    end

    assert_raise ArgumentError, ~r/threshold/, fn ->
      Extraction.extract_entities(model, "Hello", ["person"], threshold: 1.1)
    end

    assert_raise ArgumentError, ~r/exceeds/, fn ->
      Extraction.extract_entities(%{model | max_sequence_length: 1}, "Hello", ["person"])
    end
  end

  test "fixed encoder padding preserves decisions", %{model: model} do
    model = %{model | sequence_length: 128}

    assert Extraction.extract_entities(
             model,
             "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
             ["company", "person"]
           ) ==
             %{"entities" => %{"company" => ["Apple"], "person" => ["Tim Cook"]}}
  end
end
