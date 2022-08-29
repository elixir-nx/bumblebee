defmodule Bumblebee.Diffusion.UNet2DConditionTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:local, "test/models/stable-diffusion-v1-1/unet"},
                 module: Bumblebee.Diffusion.UNet2DCondition,
                 architecture: :base
               )

      assert %Bumblebee.Diffusion.UNet2DCondition{architecture: :base} = config

      inputs = %{
        "sample" => Nx.broadcast(1.0, {1, 4, 64, 64}),
        "timestep" => Nx.tensor(1),
        "encoder_hidden_states" => Nx.broadcast(0.5, {1, 1, 768})
      }

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output) == {1, 4, 64, 64}

      assert_all_close(
        output[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[-0.0082, 0.1048, 0.0681], [-0.1474, 0.0970, -0.0749], [-0.2376, 0.1794, -0.0202]],
            [[-0.8635, -0.2525, -0.2620], [-0.2213, 0.2815, 0.2756], [-0.4249, 0.0867, 0.0817]],
            [
              [-0.7380, -0.5853, -0.5205],
              [-0.4254, -0.1790, -0.0712],
              [-0.5787, -0.3730, -0.2657]
            ],
            [[-0.0085, 0.0390, 0.1853], [0.0339, 0.0459, 0.1654], [0.0093, 0.1038, 0.2129]]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
