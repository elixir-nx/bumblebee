defmodule Bumblebee.Vision.VitFeaturizerTest do
  use ExUnit.Case, async: true

  describe "integration" do
    test "encoding model input" do
      assert {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "google/vit-base-patch16-224"})

      assert %Bumblebee.Vision.VitFeaturizer{} = featurizer

      image = Nx.tensor([[50, 100], [150, 200]]) |> Nx.broadcast({3, 2, 2})

      input = Bumblebee.apply_featurizer(featurizer, image)

      assert Nx.shape(input["pixel_values"]) == {1, 3, 224, 224}
    end
  end
end
