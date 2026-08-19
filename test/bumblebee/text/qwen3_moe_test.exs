defmodule Bumblebee.Text.Qwen3MoeTest do
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
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3MoeModel"})

    assert %Bumblebee.Text.Qwen3Moe{architecture: :base} = spec
    assert spec.num_experts == 8
    # The first block uses a regular feed-forward network
    assert spec.dense_blocks == [0]

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.1923, -1.0370, 1.2795], [0.5220, 0.8177, 0.7725], [-0.9730, 0.8149, 1.9270]]
      ])
    )

    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[0.5548, -2.0859, 1.1376], [-1.0185, 0.7256, 0.7030], [-1.4676, -2.3882, 0.2991]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3MoeForCausalLM"})

    assert %Bumblebee.Text.Qwen3Moe{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0001, 0.0158, 0.0343], [-0.0615, 0.1408, 0.1419], [-0.1681, -0.1254, 0.0526]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Qwen3MoeForSequenceClassification"}
             )

    assert %Bumblebee.Text.Qwen3Moe{architecture: :for_sequence_classification} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(outputs.logits, Nx.tensor([[-0.1251, 0.0406]]))
  end
end
