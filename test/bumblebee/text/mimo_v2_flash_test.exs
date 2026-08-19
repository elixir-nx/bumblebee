defmodule Bumblebee.Text.MiMoV2FlashTest do
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
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-MiMoV2FlashModel"})

    assert %Bumblebee.Text.MiMoV2Flash{architecture: :base} = spec
    assert spec.num_experts == 8
    # The value head is smaller than the query and key heads
    assert spec.attention_head_size == 12
    assert spec.value_head_size == 8
    # The first block uses a regular feed-forward network
    assert spec.block_mlp_types == [:dense, :sparse, :sparse, :sparse]
    # The sliding window blocks use attention sinks
    assert spec.block_types == [
             :full_attention,
             :sliding_attention,
             :sliding_attention,
             :full_attention
           ]

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.9893, -0.2495, -1.0281], [-0.2055, 0.6635, 0.7067], [0.8229, 0.5511, 1.3269]]
      ])
    )

    # Positions past the sliding window
    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[-0.9767, 0.8785, -0.7310], [-0.3339, -0.3863, -0.0715], [-0.1889, 1.7892, 1.7556]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-MiMoV2FlashForCausalLM"})

    assert %Bumblebee.Text.MiMoV2Flash{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.2284, -0.1906, 0.0182], [0.1327, 0.0143, 0.1453], [0.0436, 0.1285, 0.1552]]
      ])
    )
  end
end
