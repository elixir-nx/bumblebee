defmodule Bumblebee.Vision.ClipVisionTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-CLIPModel"},
               module: Bumblebee.Vision.ClipVision,
               architecture: :base
             )

    assert %Bumblebee.Vision.ClipVision{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 226, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.4483, 0.3736, -0.5581], [0.9376, -0.3424, -0.1002], [0.5782, 0.1069, -0.2953]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.5059, 0.7391, 0.9252]])
    )
  end

  test ":for_embedding" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-CLIPModel"},
               module: Bumblebee.Vision.ClipVision,
               architecture: :for_embedding
             )

    assert %Bumblebee.Vision.ClipVision{architecture: :for_embedding} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.embedding) == {1, 64}

    assert_all_close(
      outputs.embedding[[.., 1..3]],
      Nx.tensor([[0.8865, -0.9042, -1.1233]])
    )
  end
end
