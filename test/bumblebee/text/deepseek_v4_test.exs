defmodule Bumblebee.Text.DeepseekV4Test do
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
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-DeepseekV4Model"})

    assert %Bumblebee.Text.DeepseekV4{architecture: :base} = spec

    assert spec.block_types == [
             :heavily_compressed_attention,
             :compressed_sparse_attention,
             :sliding_attention,
             :compressed_sparse_attention
           ]

    assert spec.mlp_block_types == [:hash_moe, :moe, :moe, :moe]

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0022, -0.3103, 1.8018], [-0.0323, -1.6259, -0.0381], [0.6257, 1.8550, 1.6739]]
      ])
    )

    # Positions beyond the sliding window, where the compressed entries
    # are the only source of long-range context
    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[0.4663, 0.7894, 1.1881], [0.0448, 0.2569, 2.1931], [-0.3059, -0.2497, 0.5474]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-DeepseekV4ForCausalLM"})

    assert %Bumblebee.Text.DeepseekV4{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1407, 0.1977, 0.0328], [0.0525, 0.0459, -0.0501], [-0.0345, -0.0768, 0.0712]]
      ])
    )
  end
end
