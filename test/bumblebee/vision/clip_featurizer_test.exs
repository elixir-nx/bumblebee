defmodule Bumblebee.Vision.ClipFeaturizerTest do
  use ExUnit.Case, async: true

  describe "integration" do
    test "encoding model input" do
      assert {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/clip-vit-base-patch32"})

      assert %Bumblebee.Vision.ClipFeaturizer{} = featurizer

      image = Nx.tensor([[[50], [100]], [[150], [200]]])

      inputs = Bumblebee.apply_featurizer(featurizer, image)

      assert Nx.shape(inputs["pixel_values"]) == {1, 224, 224, 3}
    end
  end
end
