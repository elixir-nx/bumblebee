defmodule Bumblebee.Diffusion.VaeKlTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "fusing/autoencoder-kl-dummy"},
                 params_filename: "diffusion_pytorch_model.bin"
               )

      assert %Bumblebee.Diffusion.VaeKl{architecture: :base} = spec

      inputs = %{
        "sample" => Nx.broadcast(0.5, {1, 3, 32, 32})
      }

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output.sample) == {1, 3, 32, 32}

      # Values from the Flax implementation using the same checkpoint
      assert_all_close(
        output.sample[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[-0.2663, -0.1856, -0.0328], [-0.3195, -0.2043, 0.0261], [-0.1438, 0.1093, -0.0887]],
            [
              [-0.1603, 0.0089, -0.0835],
              [-0.2721, -0.2134, -0.2161],
              [-0.2256, -0.4391, -0.0874]
            ],
            [[-0.1968, -0.1537, 0.0143], [-0.0999, -0.1269, -0.0189], [-0.0567, 0.1446, 0.055]]
          ]
        ]),
        atol: 1.0e-4
      )

      # Values from the PyTorch implementation with relaxed tolerance.
      # This is expected, because the 2D convolution (conv_in) gives
      # slightly different values
      assert_all_close(
        output.sample[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[-0.2663, -0.1856, -0.0329], [-0.3195, -0.2043, 0.0261], [-0.1437, 0.1092, -0.0886]],
            [
              [-0.1602, 0.0089, -0.0834],
              [-0.2720, -0.2133, -0.2161],
              [-0.2255, -0.4390, -0.0873]
            ],
            [[-0.1968, -0.1538, 0.0143], [-0.0999, -0.1270, -0.0190], [-0.0566, 0.1445, 0.0548]]
          ]
        ]),
        atol: 2.0e-4
      )
    end

    test "decoder model" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "fusing/autoencoder-kl-dummy"},
                 architecture: :decoder,
                 params_filename: "diffusion_pytorch_model.bin"
               )

      assert %Bumblebee.Diffusion.VaeKl{architecture: :decoder} = spec

      inputs = %{
        "sample" => Nx.broadcast(0.5, {1, 4, 16, 16})
      }

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output.sample) == {1, 3, 32, 32}

      assert_all_close(
        output.sample[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[-0.3571, -0.2580, -0.0133], [-0.0827, 0.0831, 0.1217], [0.8464, 0.5589, 0.0858]],
            [[-0.4579, -0.0463, 0.0853], [-0.8820, 0.0898, -0.4705], [-0.8381, -0.5012, 0.2303]],
            [[0.2384, 1.0047, 0.4958], [-0.1108, 0.4506, 0.2563], [0.2548, 0.5310, -0.2233]]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "encoder model" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "fusing/autoencoder-kl-dummy"},
                 architecture: :encoder,
                 params_filename: "diffusion_pytorch_model.bin"
               )

      assert %Bumblebee.Diffusion.VaeKl{architecture: :encoder} = spec

      inputs = %{
        "sample" => Nx.broadcast(0.5, {1, 3, 32, 32})
      }

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output.latent_dist.mean) == {1, 4, 16, 16}
      assert Nx.shape(output.latent_dist.var) == {1, 4, 16, 16}
      assert Nx.shape(output.latent_dist.logvar) == {1, 4, 16, 16}
      assert Nx.shape(output.latent_dist.std) == {1, 4, 16, 16}

      assert_all_close(
        output.latent_dist.mean[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[0.1872, 0.4903, 0.1864], [0.0671, 0.5873, 0.1105], [-0.1166, 0.2500, 0.1097]],
            [[0.2362, 0.5435, 0.2865], [-0.0456, 0.5072, 0.0343], [0.0375, 0.4808, 0.1607]],
            [[-0.0504, -0.0917, 0.0713], [0.1328, -0.0544, 0.2171], [0.3996, 0.2134, 0.1796]],
            [[0.2317, -0.1167, 0.1082], [0.4584, 0.0792, 0.0767], [0.2208, -0.0846, 0.0651]]
          ]
        ]),
        atol: 1.0e-4
      )

      assert_all_close(
        output.latent_dist.var[[0..-1//1, 0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [[1.5876, 1.0834, 1.4341], [1.7221, 1.0370, 1.2434], [1.2043, 0.8315, 1.2684]],
            [[1.6400, 0.9540, 1.4241], [2.1689, 1.1963, 1.4273], [1.5476, 0.9472, 1.3265]],
            [[0.5249, 0.6610, 0.6645], [0.4862, 0.4959, 0.6945], [0.6391, 0.7181, 0.6905]],
            [[0.8795, 1.1088, 1.2060], [1.0547, 0.9093, 0.9656], [1.0600, 0.9056, 1.1402]]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
