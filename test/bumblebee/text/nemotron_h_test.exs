defmodule Bumblebee.Text.NemotronHTest do
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
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-NemotronHModel"})

    assert %Bumblebee.Text.NemotronH{architecture: :base} = spec
    # Each block has a single mixer, of one of the four kinds
    assert spec.block_types == [:mamba, :moe, :attention, :mlp]
    assert spec.num_experts == 8

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.5597, -1.3224, -1.2639], [-1.2128, -1.6424, -0.2405], [-0.4878, -2.3987, 0.7332]]
      ])
    )

    # Positions past the state-space chunk boundary
    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[0.1792, -1.1296, -0.2379], [-0.9539, 0.3955, 0.0335], [1.3872, 0.0303, 1.0347]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-NemotronHForCausalLM"})

    assert %Bumblebee.Text.NemotronH{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0613, 0.1841, 0.0391], [-0.0759, 0.1903, -0.0652], [-0.1445, 0.1713, -0.0034]]
      ])
    )
  end
end
