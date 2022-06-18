defmodule Bumblebee.Vision.ConvNextTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :capture_log
    @tag :slow
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/convnext-tiny-224"}, architecture: :base)

      assert %Bumblebee.Vision.ConvNext{architecture: :base} = config

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.pooler_output) == {1, 768}

      assert_all_close(
        Nx.sum(output.pooler_output),
        Nx.tensor(-2.1095),
        atol: 1.0e-4
      )
    end

    @tag :slow
    test "image classification model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/convnext-tiny-224"})

      assert %Bumblebee.Vision.ConvNext{architecture: :for_image_classification} = config

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.logits) == {1, 1000}

      assert_all_close(
        output.logits[[0, 0..2]],
        Nx.tensor([-0.4239, -0.2082, 0.0709]),
        atol: 1.0e-4
      )
    end
  end
end
