defmodule Bumblebee.Vision.ResNetTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :slow
    @tag :capture_log
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "microsoft/resnet-50"}, architecture: :base)

      assert %Bumblebee.Vision.ResNet{architecture: :base} = config

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.pooler_output) == {1, 2048, 1, 1}

      assert_all_close(
        Nx.sum(output.pooler_output),
        Nx.tensor(14.5119),
        atol: 1.0e-4
      )
    end

    @tag :slow
    test "image classification model" do
      assert {:ok, model, params, config} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})

      assert %Bumblebee.Vision.ResNet{architecture: :for_image_classification} = config

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.logits) == {1, 1000}

      assert_all_close(
        output.logits[[0, 0..2]],
        Nx.tensor([-6.6223, -6.2090, -5.8592]),
        atol: 1.0e-4
      )
    end
  end
end
