defmodule Bumblebee.Conversion.PyTorchParamsTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.Conversion.PyTorchParams

  @dir Path.expand("../../fixtures/pytorch", __DIR__)

  describe "load_params!/3" do
    defp base_model() do
      Axon.input("input", shape: {nil, 4, 4, 3})
      |> Axon.conv(2, kernel_size: 2, name: "conv")
    end

    defp full_model() do
      Axon.input("input", shape: {nil, 4, 4, 3})
      |> Axon.conv(2, kernel_size: 2, name: "base.conv")
      |> Axon.flatten()
      |> Axon.dense(2, name: "classifier.intermediate")
      |> Axon.dense(1, name: "classifier.output")
    end

    defp input_template() do
      Nx.broadcast(1, {1, 4, 4, 3})
    end

    defp params_mapping() do
      %{
        "base.conv" => "base.conv",
        "classifier.intermediate" => "classifier.layers.0",
        "classifier.output" => "classifier.layers.1"
      }
    end

    test "silently loads parameters if all match" do
      model = base_model()
      path = Path.join(@dir, "state_dict_base.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          params =
            PyTorchParams.load_params!(model, input_template(), path,
              params_mapping: params_mapping()
            )

          assert_equal(params["conv"]["kernel"], Nx.broadcast(1.0, {2, 2, 3, 2}))
          assert_equal(params["conv"]["bias"], Nx.broadcast(0.0, {2}))
        end)

      refute log =~ "parameters"
    end

    test "logs parameters diff" do
      model = full_model()
      path = Path.join(@dir, "state_dict_full.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          PyTorchParams.load_params!(model, input_template(), path,
            params_mapping: params_mapping()
          )
        end)

      assert log =~ """
             the following parameters were missing:

               * classifier.output.kernel
               * classifier.output.bias
             """

      assert log =~ """
             the following PyTorch parameters were unused:

               * extra.weight
             """

      assert log =~ """
             the following parameters were ignored, because of non-matching shape:

               * classifier.intermediate.kernel (expected {18, 2}, got: {1, 1})
               * classifier.intermediate.bias (expected {2}, got: {1})
             """
    end

    test "loads parameters without prefix into a specialised model" do
      model = base_model()
      path = Path.join(@dir, "state_dict_full.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          params =
            PyTorchParams.load_params!(model, input_template(), path,
              params_mapping: params_mapping()
            )

          assert_equal(params["conv"]["kernel"], Nx.broadcast(1.0, {2, 2, 3, 2}))
          assert_equal(params["conv"]["bias"], Nx.broadcast(0.0, {2}))
        end)

      refute log =~ "conv"
    end

    test "loads parameters with prefix into a base model" do
      model = full_model()
      path = Path.join(@dir, "state_dict_base.zip.pt")

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          params =
            PyTorchParams.load_params!(model, input_template(), path,
              params_mapping: params_mapping()
            )

          assert_equal(params["base.conv"]["kernel"], Nx.broadcast(1.0, {2, 2, 3, 2}))
          assert_equal(params["base.conv"]["bias"], Nx.broadcast(0.0, {2}))
        end)

      refute log =~ "conv"
    end
  end
end
