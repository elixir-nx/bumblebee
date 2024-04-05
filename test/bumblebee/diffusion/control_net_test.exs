defmodule Bumblebee.Diffusion.ControlNetTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "lllyasviel/sd-controlnet-scribble"},
               module: Bumblebee.Diffusion.ControlNet,
               architecture: :base
             )

    assert %Bumblebee.Diffusion.ControlNet{
             architecture: :base
           } = spec

    inputs = %{
      "sample" => Nx.broadcast(0.5, {1, 64, 64, 4}),
      "conditioning" => Nx.broadcast(0.8, {1, 512, 512, 3}),
      "timestep" => Nx.tensor(0),
      "encoder_hidden_state" => Nx.broadcast(0.8, {1, 1, 768})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.mid_block_residual) == {1, 8, 8, 1280}

    assert_all_close(
      outputs.mid_block_residual[[0, 0, 0, 1..3]],
      Nx.tensor([-1.2827045917510986, -0.6995724439620972, -0.610561192035675])
    )

    first_down_residual = elem(outputs.down_blocks_residuals, 0)
    assert Nx.shape(first_down_residual) == {1, 64, 64, 320}

    assert_all_close(
      first_down_residual[[0, 0, 0, 1..3]],
      Nx.tensor([-0.029463158920407295, 0.04885300621390343, -0.12834328413009644])
    )
  end
end
