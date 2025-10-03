defmodule Bumblebee.Text.SmolLM3Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "joelkoch/tiny_random_smollm3"}, architecture: :base)

    assert %Bumblebee.Text.SmolLM3{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 64}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.4167, -0.0137, 0.7160], [-0.2624, -1.1185, -0.3098], [-0.0383, -0.8390, -0.0039]]
      ])
    )
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "joelkoch/tiny_smollm3_for_question_answering"},
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
        [0.0656, 0.0358, -0.0395, 0.0227, 0.0594, 0.0942, -0.2356, 0.0244, 0.0701, 0.0705]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "joelkoch/tiny_smollm3_for_sequence_classification"},
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
      Nx.tensor([[0.1468, -0.0980]])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "joelkoch/tiny_random_smollm3"},
               architecture: :for_causal_language_modeling
             )

    assert %Bumblebee.Text.SmolLM3{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 128_256}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0602, 0.1254, 0.0077], [0.0187, 0.0270, 0.0625], [-0.0079, -0.0478, 0.1872]]
      ])
    )
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "joelkoch/tiny_smollm3_for_token_classification"},
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
          [-0.1358, 0.1047],
          [-0.0504, 0.1214],
          [0.1960, -0.0031],
          [0.0428, 0.0429],
          [-0.0680, 0.1391],
          [0.0828, 0.0945],
          [-0.0144, -0.2466],
          [0.0152, 0.1096],
          [0.1437, -0.1766],
          [0.1439, -0.1762]
        ]
      ])
    )
  end
end
