defmodule Bumblebee.Text.LagunaTest do
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
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-LagunaModel"})

    assert %Bumblebee.Text.Laguna{architecture: :base} = spec
    assert spec.num_experts == 8
    # The first block uses a regular feed-forward network
    assert spec.block_mlp_types == [:dense, :sparse, :sparse, :sparse]
    # The blocks alternate between full and sliding window attention,
    # with a different number of heads
    assert spec.block_types == [
             :full_attention,
             :sliding_attention,
             :sliding_attention,
             :full_attention
           ]

    assert spec.block_num_attention_heads == [4, 6, 6, 4]

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[1.2983, -2.1187, 1.5614], [1.6172, -0.5286, -0.3094], [-0.7904, -0.4496, 0.1295]]
      ])
    )

    # Positions past the sliding window
    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[-0.2473, -1.5746, 2.2151], [0.3197, 0.2362, -0.0956], [-0.0552, 0.7520, 1.1032]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-LagunaForCausalLM"})

    assert %Bumblebee.Text.Laguna{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0695, 0.1079, 0.0906], [0.0266, -0.0100, 0.1340], [-0.0429, -0.1877, 0.1798]]
      ])
    )
  end
end
