defmodule Bumblebee.Text.Qwen3Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "tiny-random/qwen3"}, architecture: :base)

    assert %Bumblebee.Text.Qwen3{architecture: :base} = spec
    assert spec.use_qk_norm == true

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 64}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [0.0437, -0.0292, 0.6567],
          [-0.0767, 0.0107, 0.2657],
          [0.4693, -0.0452, 0.2521]
        ]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "tiny-random/qwen3"})

    assert %Bumblebee.Text.Qwen3{architecture: :for_causal_language_modeling} = spec
    assert spec.use_qk_norm == true

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 151936}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [2.5975, 3.9118, -0.7135],
          [1.8620, 0.6854, 2.3352],
          [0.9874, -4.0238, -0.1917]
        ]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "tiny-random/qwen3"},
               architecture: :for_sequence_classification
             )

    assert %Bumblebee.Text.Qwen3{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    # Note: tiny-random model is missing sequence_classification_head parameters,
    # so it uses random initialization. We only verify the shape is correct.
    assert Nx.shape(outputs.logits) == {1, 2}
  end

  test ":for_embedding" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "tiny-random/qwen3"}, architecture: :for_embedding)

    assert %Bumblebee.Text.Qwen3{architecture: :for_embedding} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.embedding) == {1, 64}

    assert_all_close(
      outputs.embedding[[.., 1..3]],
      Nx.tensor([[0.2217, -0.0037, -0.1757]])
    )
  end
end
