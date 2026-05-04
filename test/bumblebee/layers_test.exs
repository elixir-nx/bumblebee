defmodule Bumblebee.LayersTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "fp8_aware_dense/3" do
    test "dequantizes FP8 kernel with scale_inv" do
      # Create a simple model with fp8_aware_dense
      model =
        Axon.input("input", shape: {nil, 4})
        |> Bumblebee.Layers.fp8_aware_dense(8, name: "dense", block_size: 2)

      # Create params with known values
      # kernel: [4, 8] - input_features x output_features
      # scale_inv: [2, 4] - ceil(4/2) x ceil(8/2) blocks
      kernel =
        Nx.tensor(
          [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8]
          ],
          type: {:f, 32}
        )

      # Scale of 2.0 for all blocks means output should be 2x what it would be without scaling
      scale_inv =
        Nx.tensor(
          [
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0]
          ],
          type: {:f, 32}
        )

      params = %{
        "dense" => %{
          "kernel" => kernel,
          "scale_inv" => scale_inv
        }
      }

      input = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])

      output = Axon.predict(model, params, %{"input" => input})

      # Without scaling: input [1,1,1,1] dot kernel gives [4, 8, 12, 16, 20, 24, 28, 32]
      # With scale_inv of 2.0: [8, 16, 24, 32, 40, 48, 56, 64]
      expected = Nx.tensor([[8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]])

      assert_all_close(output, expected)
    end

    test "dequantizes with identity scale (1.0)" do
      model =
        Axon.input("input", shape: {nil, 4})
        |> Bumblebee.Layers.fp8_aware_dense(4, name: "dense", block_size: 2)

      kernel =
        Nx.tensor(
          [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
          ],
          type: {:f, 32}
        )

      # Identity scale
      scale_inv =
        Nx.tensor(
          [
            [1.0, 1.0],
            [1.0, 1.0]
          ],
          type: {:f, 32}
        )

      params = %{
        "dense" => %{
          "kernel" => kernel,
          "scale_inv" => scale_inv
        }
      }

      input = Nx.tensor([[2.0, 3.0, 4.0, 5.0]])
      output = Axon.predict(model, params, %{"input" => input})

      # Identity matrix with scale 1.0 should return input unchanged
      assert_all_close(output, input)
    end

    test "handles non-block-aligned dimensions" do
      # 3 input features, 5 output features with block_size 2
      # This tests the slicing logic for non-aligned dimensions
      model =
        Axon.input("input", shape: {nil, 3})
        |> Bumblebee.Layers.fp8_aware_dense(5, name: "dense", block_size: 2)

      # kernel: [3, 5]
      kernel = Nx.broadcast(1.0, {3, 5})

      # scale_inv: [ceil(3/2), ceil(5/2)] = [2, 3]
      scale_inv = Nx.broadcast(1.0, {2, 3})

      params = %{
        "dense" => %{
          "kernel" => kernel,
          "scale_inv" => scale_inv
        }
      }

      input = Nx.tensor([[1.0, 1.0, 1.0]])
      output = Axon.predict(model, params, %{"input" => input})

      # Sum of 3 ones = 3.0 for each output
      expected = Nx.tensor([[3.0, 3.0, 3.0, 3.0, 3.0]])

      assert_all_close(output, expected)
    end

    test "includes bias when use_bias is true" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Bumblebee.Layers.fp8_aware_dense(2, name: "dense", block_size: 2, use_bias: true)

      kernel =
        Nx.tensor(
          [
            [1, 0],
            [0, 1]
          ],
          type: {:f, 32}
        )

      scale_inv = Nx.tensor([[1.0]], type: {:f, 32})
      bias = Nx.tensor([10.0, 20.0], type: {:f, 32})

      params = %{
        "dense" => %{
          "kernel" => kernel,
          "scale_inv" => scale_inv,
          "bias" => bias
        }
      }

      input = Nx.tensor([[1.0, 2.0]])
      output = Axon.predict(model, params, %{"input" => input})

      # [1, 2] with identity kernel = [1, 2], plus bias [10, 20] = [11, 22]
      expected = Nx.tensor([[11.0, 22.0]])

      assert_all_close(output, expected)
    end
  end
end
