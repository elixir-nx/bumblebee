defmodule Bumblebee.Vision.BlipVisionTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BlipModel"},
               module: Bumblebee.Vision.BlipVision,
               architecture: :base
             )

    assert %Bumblebee.Vision.BlipVision{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 226, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]] |> Nx.multiply(1_000_000),
      Nx.tensor([
        [[-0.0272, -0.0129, 0.0174], [0.0069, -0.0429, -0.0334], [0.0428, -0.0797, -0.0353]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]] |> Nx.multiply(10_000),
      Nx.tensor([[-0.0128, -0.0792, -0.1011]])
    )
  end
end
