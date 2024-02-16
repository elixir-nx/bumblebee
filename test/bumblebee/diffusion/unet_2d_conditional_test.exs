defmodule Bumblebee.Diffusion.UNet2DConditionalTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-stable-diffusion-torch", subdir: "unet"}
             )

    assert %Bumblebee.Diffusion.UNet2DConditional{architecture: :base} = spec

    inputs = %{
      "sample" => Nx.broadcast(0.5, {1, 32, 32, 4}),
      "timestep" => Nx.tensor(1),
      "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 32})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.sample) == {1, 32, 32, 4}

    assert_all_close(
      outputs.sample[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [[-1.0813, -0.2179, 1.3359], [-0.5109, -0.2799, 0.8373], [-0.1545, -1.0922, -0.2392]],
          [[-0.8094, -0.9485, 0.9448], [-1.2588, -0.8376, -0.0478], [-0.8355, 0.0843, 0.6881]],
          [[-0.9218, -0.9650, -0.0154], [-1.2142, -0.7105, -0.5304], [-0.6982, -0.3920, 0.2081]]
        ]
      ])
    )
  end
end
