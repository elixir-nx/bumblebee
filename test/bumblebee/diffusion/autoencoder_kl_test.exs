defmodule Bumblebee.Diffusion.AutoencoderKlTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:local, "test/models/stable-diffusion-v1-1/vae"},
                 module: Bumblebee.Diffusion.AutoencoderKl,
                 architecture: :base
               )

      assert %Bumblebee.Diffusion.AutoencoderKl{architecture: :base} = config

      inputs = %{
        "sample" => Nx.broadcast(1.0, {1, 3, 64, 64})
      }

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output) == {1, 3, 64, 64}

      assert_all_close(
        output[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[0.9500, 0.9958, 0.9803], [0.9750, 0.9885, 0.9890], [0.9742, 0.9922, 0.9866]],
            [[0.9897, 0.9993, 1.0007], [0.9864, 0.9818, 0.9850], [0.9914, 0.9816, 0.9843]],
            [[1.0103, 1.0158, 1.0072], [1.0172, 0.9994, 1.0098], [0.9975, 0.9967, 0.9968]]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "decoder model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:local, "test/models/stable-diffusion-v1-1/vae"},
                 module: Bumblebee.Diffusion.AutoencoderKl,
                 architecture: :decoder
               )

      assert %Bumblebee.Diffusion.AutoencoderKl{architecture: :decoder} = config

      inputs = %{
        "sample" => Nx.broadcast(0.5, {1, 4, 8, 8})
      }

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output) == {1, 3, 64, 64}

      assert_all_close(
        output[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[0.0103, 0.0136, -0.0209], [0.0415, -0.0183, 0.0083], [-0.0392, -0.0089, 0.0117]],
            [
              [-0.0814, -0.0744, -0.0992],
              [-0.0842, -0.0776, -0.0877],
              [-0.0868, -0.0845, -0.0522]
            ],
            [
              [-0.2444, -0.2644, -0.2924],
              [-0.2427, -0.2991, -0.3028],
              [-0.2596, -0.2850, -0.2729]
            ]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
