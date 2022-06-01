defmodule Bumblebee.Conversion.PyTorchTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.Conversion.PyTorch

  @dir Path.expand("../fixtures/pytorch", __DIR__)

  describe "load_params!/3" do
    defp base_model() do
      Axon.input({nil, 3, 4, 4})
      |> Axon.conv(2, kernel_size: 2, name: "conv")
    end

    defp full_model() do
      Axon.input({nil, 3, 4, 4})
      |> Axon.conv(2, kernel_size: 2, name: "base.conv")
      |> Axon.flatten()
      |> Axon.dense(2, name: "classifier.layers.0")
      |> Axon.dense(1, name: "classifier.layers.1")
    end

    test "silently loads parameters if all match" do
      model = base_model()
      path = Path.join(@dir, "state_dict_base.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          params = PyTorch.load_params!(model, path, base_model_prefix: "base")

          assert_equal(params["conv"]["kernel"], Nx.broadcast(1.0, {2, 3, 2, 2}))
          assert_equal(params["conv"]["bias"], Nx.broadcast(0.0, {2}))
        end)

      assert log == ""
    end

    test "logs parameters diff" do
      model = full_model()
      path = Path.join(@dir, "state_dict_full.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          PyTorch.load_params!(model, path)
        end)

      assert log =~ """
             the following parameters were missing:

               * classifier.layers.1.kernel
               * classifier.layers.1.bias
             """

      assert log =~ """
             the following PyTorch parameters were unused:

               * extra.weight
             """

      assert log =~ """
             the following parameters were ignored, because of non-matching shape:

               * classifier.layers.0.kernel (expected {18, 2}, got: {1, 1})
               * classifier.layers.0.bias (expected {2}, got: {1})
             """
    end

    test "loads parameters without prefix into a specialised model" do
      model = base_model()
      path = Path.join(@dir, "state_dict_full.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          params = PyTorch.load_params!(model, path, base_model_prefix: "base")

          assert_equal(params["conv"]["kernel"], Nx.broadcast(1.0, {2, 3, 2, 2}))
          assert_equal(params["conv"]["bias"], Nx.broadcast(0.0, {2}))
        end)

      refute log =~ "conv"
    end

    test "loads parameters with prefix into a base model" do
      model = full_model()
      path = Path.join(@dir, "state_dict_base.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          params = PyTorch.load_params!(model, path, base_model_prefix: "base")

          assert_equal(params["base.conv"]["kernel"], Nx.broadcast(1.0, {2, 3, 2, 2}))
          assert_equal(params["base.conv"]["bias"], Nx.broadcast(0.0, {2}))
        end)

      refute log =~ "conv"
    end
  end
end
