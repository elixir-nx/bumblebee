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
          [0.256240, -0.424804, -0.137127],
          [-0.806056, -0.141523, 0.364655],
          [-0.407146, -1.018769, -1.137962]
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
          0.193774000,
          -0.034538623,
          0.091337912,
          -0.082132638,
          -0.065818049,
          -0.043813121,
          -0.052513212,
          -0.077175386,
          -0.127072141,
          -0.127050534
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
      Nx.tensor([[-0.056712, 0.024943]]),
      atol: 1.0e-3
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
          [-0.043879, 0.297687, 0.132621],
          [0.028554, 0.049315, 0.053559],
          [0.045794, 0.230346, 0.085419]
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
          [-0.005348, -0.063641],
          [-0.125814, 0.258190],
          [-0.050098, 0.048529],
          [-0.113691, -0.065939],
          [0.042302, 0.130302],
          [0.080049, 0.074302],
          [-0.137889, 0.070994],
          [-0.032226, 0.148868],
          [-0.091657, -0.029607],
          [-0.091757, -0.029316]
        ]
      ]),
      atol: 1.0e-3
    )
  end
end
