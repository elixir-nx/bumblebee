defmodule Bumblebee.Diffusion.UNet2DConditionalTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model(
                 {:hf, "hakurei/waifu-diffusion", subdir: "unet"},
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
        to_channels_first(outputs.sample)[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [
              [-0.0990, -0.1132, -0.0237],
              [-0.1776, -0.1141, -0.0475],
              [-0.1518, -0.0989, -0.0500]
            ],
            [[-0.3094, 0.1087, 0.1059], [-0.1391, 0.2548, 0.2616], [-0.1631, 0.1674, 0.1968]],
            [
              [-0.6858, -0.4824, -0.3525],
              [-0.3155, -0.1462, -0.0410],
              [-0.3399, -0.2399, -0.0982]
            ],
            [[-0.0407, 0.0768, 0.1296], [-0.0119, 0.1665, 0.2012], [0.0084, 0.2265, 0.2371]]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
