defmodule Bumblebee.Vision.ResNetTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "microsoft/resnet-50"}, architecture: :base)

      assert %Bumblebee.Vision.ResNet{architecture: :base} = spec

      inputs = %{"pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})}
      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.pooled_state) == {1, 1, 1, 2048}

      assert_all_close(
        Nx.sum(outputs.pooled_state),
        Nx.tensor(14.5119),
        atol: 1.0e-4
      )
    end

    test "image classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "microsoft/resnet-50"})

      assert %Bumblebee.Vision.ResNet{architecture: :for_image_classification} = spec

      inputs = %{"pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})}
      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 1000}

      assert_all_close(
        outputs.logits[[0, 0..2]],
        Nx.tensor([-6.6223, -6.2090, -5.8592]),
        atol: 1.0e-4
      )
    end
  end
end
