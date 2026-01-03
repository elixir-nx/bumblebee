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
      "input_ids" => Nx.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 6, 768}
    assert Nx.shape(outputs.pooled_state) == {1, 768}

    # Values verified against Python transformers
    assert_all_close(
      outputs.hidden_state[[.., 0, 0..4]],
      Nx.tensor([[1.3752, 0.7431, -4.6988, -0.6574, 2.1887]]),
      atol: 1.0e-3
    )

    assert_all_close(
      outputs.pooled_state[[.., 0..4]],
      Nx.tensor([[1.0917, 0.5968, -3.9347, -0.6988, 1.5423]]),
      atol: 1.0e-3
    )
  end
end
