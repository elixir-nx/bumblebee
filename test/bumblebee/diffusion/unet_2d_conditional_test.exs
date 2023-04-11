defmodule Bumblebee.Diffusion.UNet2DConditionalTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "CompVis/stable-diffusion-v1-4", subdir: "unet"},
                 params_filename: "diffusion_pytorch_model.bin"
               )

      assert %Bumblebee.Diffusion.UNet2DConditional{architecture: :base} = spec

      inputs = %{
        "sample" => Nx.broadcast(0.5, {1, 32, 32, 4}),
        "timestep" => Nx.tensor(1),
        "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 768})
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.sample) == {1, 32, 32, 4}

      assert_all_close(
        to_channels_first(outputs.sample)[[.., .., 1..3, 1..3]],
        Nx.tensor([
          [
            [[0.0283, -0.0525, 0.0433], [-0.1055, -0.1024, -0.0299], [-0.0498, -0.0391, 0.0032]],
            [[-0.2615, 0.1989, 0.1763], [-0.1742, 0.2385, 0.2464], [-0.2188, 0.1589, 0.1809]],
            [
              [-0.5708, -0.3721, -0.2976],
              [-0.2256, -0.0616, -0.0092],
              [-0.2484, -0.1358, -0.0635]
            ],
            [[0.0672, 0.2093, 0.2373], [0.0086, 0.1947, 0.2024], [0.0041, 0.1981, 0.2100]]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
