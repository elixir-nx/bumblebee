defmodule Bumblebee.Vision.ConvNextFeaturizerTest do
  use ExUnit.Case, async: true

  describe "integration" do
    test "encoding model input" do
      assert {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "facebook/convnext-tiny-224"})

      assert %Bumblebee.Vision.ConvNextFeaturizer{} = featurizer

      image = Nx.tensor([[[50], [100]], [[150], [200]]])

      input = Bumblebee.apply_featurizer(featurizer, image)

      assert Nx.shape(input["pixel_values"]) == {1, 224, 224, 3}
    end
  end
end
