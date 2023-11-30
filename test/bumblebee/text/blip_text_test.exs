defmodule Bumblebee.Text.BlipTextTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BlipModel"},
               module: Bumblebee.Text.BlipText,
               architecture: :base
             )

    assert %Bumblebee.Text.BlipText{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.9281, 1.2373, 0.4223], [-1.1549, 2.1187, -0.9194], [0.0237, -0.7517, 0.5720]]
      ]),
      atol: 1.0e-4
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BlipForConditionalGeneration"},
               module: Bumblebee.Text.BlipText,
               architecture: :for_causal_language_modeling
             )

    assert %Bumblebee.Text.BlipText{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1124}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0736, -0.0142, 0.2178], [0.0744, 0.0990, 0.1510], [-0.1186, -0.1449, -0.0643]]
      ]),
      atol: 1.0e-4
    )
  end
end
