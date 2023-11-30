defmodule Bumblebee.Text.ClipTextTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-CLIPModel"},
               module: Bumblebee.Text.ClipText,
               architecture: :base
             )

    assert %Bumblebee.Text.ClipText{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1696, -0.2324, -0.1659], [-0.0525, -0.3103, 0.1557], [-0.2566, -0.4519, 0.6398]]
      ]),
      atol: 1.0e-4
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.6903, -1.2524, 1.5328]]),
      atol: 1.0e-4
    )
  end

  test ":for_embedding" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-CLIPModel"},
               module: Bumblebee.Text.ClipText,
               architecture: :for_embedding
             )

    assert %Bumblebee.Text.ClipText{architecture: :for_embedding} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.embedding) == {1, 64}

    assert_all_close(
      outputs.embedding[[.., 1..3]],
      Nx.tensor([[1.1069, -0.0839, -1.6185]]),
      atol: 1.0e-4
    )
  end
end
