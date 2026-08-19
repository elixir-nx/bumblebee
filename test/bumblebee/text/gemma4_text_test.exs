defmodule Bumblebee.Text.Gemma4TextTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  defp inputs() do
    %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }
  end

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Gemma4TextModel"})

    assert %Bumblebee.Text.Gemma4Text{architecture: :base} = spec
    assert spec.num_experts == nil

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0493, -0.3482, 0.8963], [0.4703, -0.2422, 0.0286], [1.5399, -0.1897, -0.4324]]
      ])
    )

    # Positions beyond the sliding window of the local attention blocks
    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[0.7115, -0.1246, 0.6040], [0.3985, 1.3864, 0.5138], [0.6804, 0.0218, -0.8041]]
      ])
    )
  end

  test ":base with mixture-of-experts blocks" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Gemma4TextMoeModel"})

    assert %Bumblebee.Text.Gemma4Text{architecture: :base} = spec
    assert spec.num_experts == 8

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-1.4655, -0.1762, -0.0451], [-1.4502, -1.1301, 0.3099], [-1.2364, -0.4069, 0.0603]]
      ])
    )

    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[-0.2601, -1.4350, 1.2006], [0.3692, -1.1644, 1.0500], [-0.0443, -1.5164, -0.3469]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Gemma4ForCausalLM"})

    assert %Bumblebee.Text.Gemma4Text{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.1582, -0.1003, -0.1688], [0.0772, 0.0410, -0.2913], [-0.1841, -0.0072, -0.0315]]
      ])
    )
  end
end
