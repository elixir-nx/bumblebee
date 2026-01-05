defmodule Bumblebee.Text.NomicBertTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @tag :slow
  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "nomic-ai/nomic-embed-text-v1.5"})

    assert %Bumblebee.Text.NomicBert{architecture: :base} = spec
    assert spec.hidden_size == 768
    assert spec.num_blocks == 12
    assert spec.num_attention_heads == 12
    assert spec.rotary_embedding_base == 1000

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 768}
    assert Nx.shape(outputs.pooled_state) == {1, 768}

    # Values verified against Python transformers
    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([[[0.0315, -5.2254, 0.0180],
                  [0.0877, -5.3772, 0.1800],
                  [-0.0546, -4.8813, 0.2614]]])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[0.0340, -5.2018, 0.1686]])
    )
  end
end
