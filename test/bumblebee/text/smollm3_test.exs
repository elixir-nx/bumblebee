defmodule Bumblebee.Text.SmolLM3Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-SmolLM3Model"},
               architecture: :base
             )

    assert %Bumblebee.Text.SmolLM3{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [0.2562, -0.4248, -0.1371],
          [-0.8060, -0.1415, 0.3646],
          [-0.4071, -1.0187, -1.1379]
        ]
      ]),
      atol: 1.0e-3,
      rtol: 1.0e-3
    )
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-SmolLM3ForQuestionAnswering"},
               architecture: :for_question_answering
             )

    assert %Bumblebee.Text.SmolLM3{architecture: :for_question_answering} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.end_logits) == {1, 10}

    assert_all_close(
      outputs.end_logits,
      Nx.tensor([
        [
          0.1937,
          -0.0345,
          0.0913,
          -0.0821,
          -0.0658,
          -0.0438,
          -0.0525,
          -0.0771,
          -0.1270,
          -0.1270
        ]
      ]),
      atol: 1.0e-3
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-SmolLM3ForSequenceClassification"},
               architecture: :for_sequence_classification
             )

    assert %Bumblebee.Text.SmolLM3{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0567, 0.0249]]),
      rtol: 1.0e-2
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-SmolLM3ForCausalLM"},
               architecture: :for_causal_language_modeling
             )

    assert %Bumblebee.Text.SmolLM3{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [-0.0438, 0.2976, 0.1326],
          [0.0285, 0.0493, 0.0535],
          [0.0457, 0.2303, 0.0854]
        ]
      ]),
      atol: 1.0e-3
    )
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-SmolLM3ForTokenClassification"},
               architecture: :for_token_classification
             )

    assert %Bumblebee.Text.SmolLM3{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([
        [
          [-0.0053, -0.0636],
          [-0.1258, 0.2581],
          [-0.0500, 0.0485],
          [-0.1136, -0.0659],
          [0.0423, 0.1303],
          [0.0800, 0.0743],
          [-0.1378, 0.0709],
          [-0.0322, 0.1488],
          [-0.0916, -0.0296],
          [-0.0917, -0.0293]
        ]
      ]),
      atol: 1.0e-3
    )
  end
end
