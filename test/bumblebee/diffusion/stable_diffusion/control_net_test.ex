defmodule Bumblebee.Diffusion.StableDiffusion.ControlNetTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @tag timeout: :infinity
  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "lllyasviel/sd-controlnet-scribble"},
               module: Bumblebee.Diffusion.StableDiffusion.ControlNet,
               architecture: :base
             )

    assert %Bumblebee.Diffusion.StableDiffusion.ControlNet{
             architecture: :base
           } = spec

    inputs = %{
      "sample" => Nx.broadcast(0.5, {1, 64, 64, 4}),
      "controlnet_conditioning" => Nx.broadcast(0.8, {1, 512, 512, 4}),
      "timestep" => Nx.tensor(1),
      "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 32})
    }

    outputs = Axon.predict(model, params, inputs, debug: true)

    assert Nx.shape(outputs.sample) == {1, 32, 32, 4}

    assert_all_close(
      to_channels_first(outputs.sample)[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [
            [-1.0813, -0.5109, -0.1545],
            [-0.8094, -1.2588, -0.8355],
            [-0.9218, -1.2142, -0.6982]
          ],
          [
            [-0.2179, -0.2799, -1.0922],
            [-0.9485, -0.8376, 0.0843],
            [-0.9650, -0.7105, -0.3920]
          ],
          [[1.3359, 0.8373, -0.2392], [0.9448, -0.0478, 0.6881], [-0.0154, -0.5304, 0.2081]]
        ]
      ]),
      atol: 1.0e-4
    )
  end
end
