defmodule Bumblebee.Vision.BlipFeaturizerTest do
  use ExUnit.Case, async: true

  test "encodes image" do
    assert {:ok, featurizer} =
             Bumblebee.load_featurizer({:hf, "Salesforce/blip-image-captioning-base"})

    assert %Bumblebee.Vision.BlipFeaturizer{} = featurizer

    image = Nx.tensor([[[50], [100]], [[150], [200]]]) |> Nx.broadcast({2, 2, 3})

    inputs = Bumblebee.apply_featurizer(featurizer, image)

    assert Nx.shape(inputs["pixel_values"]) == {1, 384, 384, 3}
  end
end
