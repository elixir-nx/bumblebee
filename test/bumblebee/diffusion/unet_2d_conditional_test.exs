defmodule Bumblebee.Diffusion.UNet2DConditionalTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model(
                 {:hf, "doohickey/trinart-waifu-diffusion-50-50", subdir: "unet"},
                 params_filename: "diffusion_pytorch_model.bin"
               )

      assert %Bumblebee.Diffusion.UNet2DConditional{architecture: :base} = config

      inputs = %{
        "sample" => Nx.broadcast(0.5, {1, 4, 32, 32}),
        "timestep" => Nx.tensor(1),
        "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 768})
      }

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output.sample) == {1, 4, 32, 32}

      assert_all_close(
        output.sample[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[0.0472, -0.0103, 0.0759], [-0.1094, -0.0668, -0.0210], [-0.0628, -0.0139, 0.0059]],
            [[-0.3422, 0.1162, 0.1118], [-0.2334, 0.2019, 0.2247], [-0.2663, 0.1356, 0.1789]],
            [
              [-0.6039, -0.4148, -0.3095],
              [-0.2547, -0.1128, -0.0326],
              [-0.2782, -0.1928, -0.0814]
            ],
            [[-0.0464, 0.0985, 0.1505], [-0.0665, 0.1422, 0.1747], [-0.0512, 0.1882, 0.2173]]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
