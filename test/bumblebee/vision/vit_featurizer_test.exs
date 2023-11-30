defmodule Bumblebee.Vision.VitFeaturizerTest do
  use ExUnit.Case, async: true

  test "encodes image" do
    assert {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "google/vit-base-patch16-224"})

    assert %Bumblebee.Vision.VitFeaturizer{} = featurizer

    image = Nx.tensor([[[50], [100]], [[150], [200]]]) |> Nx.broadcast({2, 2, 3})

    inputs = Bumblebee.apply_featurizer(featurizer, image)

    assert Nx.shape(inputs["pixel_values"]) == {1, 224, 224, 3}
  end
end
