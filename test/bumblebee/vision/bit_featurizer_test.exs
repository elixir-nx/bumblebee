defmodule Bumblebee.Vision.BitFeaturizerTest do
  use ExUnit.Case, async: true

  test "encodes image" do
    assert {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "google/bit-50"})

    assert %Bumblebee.Vision.BitFeaturizer{} = featurizer

    image = Nx.tensor([[[50], [100]], [[150], [200]]]) |> Nx.broadcast({2, 2, 3})

    inputs = Bumblebee.apply_featurizer(featurizer, image)

    assert Nx.shape(inputs["pixel_values"]) == {1, 448, 448, 3}
  end
end
