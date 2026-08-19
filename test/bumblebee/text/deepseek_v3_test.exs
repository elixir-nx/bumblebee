defmodule Bumblebee.Text.DeepseekV3Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-DeepseekV3Model"})

    assert %Bumblebee.Text.DeepseekV3{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-1.0463, 0.1638, 0.3528], [1.1179, -0.5842, 0.6826], [-1.0969, -0.0950, 1.1230]]
      ])
    )

    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[0.7282, -1.0314, -0.3837], [-1.3051, -2.7814, 1.0082], [-1.4544, 0.6658, -0.8316]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-DeepseekV3ForCausalLM"})

    assert %Bumblebee.Text.DeepseekV3{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.2467, 0.0028, -0.0160], [-0.2260, -0.0206, 0.1620], [-0.0605, -0.0472, -0.0417]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-DeepseekV3ForSequenceClassification"}
             )

    assert %Bumblebee.Text.DeepseekV3{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(outputs.logits, Nx.tensor([[-0.2290, 0.0165]]))
  end
end
