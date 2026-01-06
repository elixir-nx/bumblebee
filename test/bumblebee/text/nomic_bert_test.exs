defmodule Bumblebee.Text.NomicBertTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-NomicBertModel"})

    assert %Bumblebee.Text.NomicBert{architecture: :base} = spec

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
        [[1.5269, -0.3709, -0.6235], [0.0301, -0.1500, -1.0316], [-1.4733, -1.1167, 0.2346]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[0.1788, -0.2985, 0.4405]])
    )
  end
end
