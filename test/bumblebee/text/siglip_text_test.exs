defmodule Bumblebee.Text.SigLipTextTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-SiglipTextModel"})

    assert %Bumblebee.Text.SigLipText{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 5, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.3263, -0.2521, 0.1006], [-0.3991, -0.1694, -1.6610], [-0.0146, -1.2169, -2.0933]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.6236, -1.1488, -2.4142]])
    )
  end
end
