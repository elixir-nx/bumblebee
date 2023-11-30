defmodule Bumblebee.Multimodal.LayoutLmTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-LayoutLMModel"})

    assert %Bumblebee.Multimodal.LayoutLm{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "bounding_box" =>
        Nx.tensor([
          [
            [10, 12, 16, 18],
            [20, 22, 26, 28],
            [30, 32, 36, 38],
            [40, 42, 46, 48],
            [50, 52, 56, 58],
            [60, 62, 66, 68],
            [70, 72, 76, 78],
            [80, 82, 86, 88],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([[[0.0240, -0.8855, 1.8877], [1.8435, 0.6223, 2.0573], [1.6961, -1.2411, 1.2824]]]),
      atol: 1.0e-4
    )
  end

  test ":for_masked_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-LayoutLMForMaskedLM"})

    assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_masked_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "bounding_box" =>
        Nx.tensor([
          [
            [10, 12, 16, 18],
            [20, 22, 26, 28],
            [30, 32, 36, 38],
            [40, 42, 46, 48],
            [50, 52, 56, 58],
            [60, 62, 66, 68],
            [70, 72, 76, 78],
            [80, 82, 86, 88],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1124}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.2101, -0.0342, 0.1613], [-0.0734, 0.1874, -0.0231], [-0.0776, 0.0145, 0.2504]]
      ]),
      atol: 1.0e-4
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-LayoutLMForSequenceClassification"}
             )

    assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "bounding_box" =>
        Nx.tensor([
          [
            [10, 12, 16, 18],
            [20, 22, 26, 28],
            [30, 32, 36, 38],
            [40, 42, 46, 48],
            [50, 52, 56, 58],
            [60, 62, 66, 68],
            [70, 72, 76, 78],
            [80, 82, 86, 88],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0241, 0.0096]]),
      atol: 1.0e-4
    )
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-LayoutLMForTokenClassification"}
             )

    assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "bounding_box" =>
        Nx.tensor([
          [
            [10, 12, 16, 18],
            [20, 22, 26, 28],
            [30, 32, 36, 38],
            [40, 42, 46, 48],
            [50, 52, 56, 58],
            [60, 62, 66, 68],
            [70, 72, 76, 78],
            [80, 82, 86, 88],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}

    assert_all_close(
      outputs.logits[[.., 1..3//1, ..]],
      Nx.tensor([[[-0.1849, 0.1134], [-0.1329, 0.0025], [-0.0454, 0.0441]]]),
      atol: 1.0e-4
    )
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-LayoutLMForQuestionAnswering"}
             )

    assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_question_answering} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "token_type_ids" => Nx.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]]),
      "bounding_box" =>
        Nx.tensor([
          [
            [10, 12, 16, 18],
            [20, 22, 26, 28],
            [30, 32, 36, 38],
            [40, 42, 46, 48],
            [50, 52, 56, 58],
            [60, 62, 66, 68],
            [70, 72, 76, 78],
            [80, 82, 86, 88],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.start_logits) == {1, 10}
    assert Nx.shape(outputs.end_logits) == {1, 10}

    assert_all_close(
      outputs.start_logits[[.., 1..3]],
      Nx.tensor([[-0.1853, 0.1580, 0.2387]]),
      atol: 1.0e-3
    )

    assert_all_close(
      outputs.end_logits[[.., 1..3]],
      Nx.tensor([[-0.1854, -0.0074, 0.0670]]),
      atol: 1.0e-3
    )
  end
end
