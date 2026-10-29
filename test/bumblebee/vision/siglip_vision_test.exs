defmodule Bumblebee.Vision.SigLipVisionTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-SiglipVisionModel"})

    assert %Bumblebee.Vision.SigLipVision{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 225, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.3204, 1.1951, 0.9654], [-0.3518, 1.4216, 0.9595], [-0.4153, 1.3504, 0.6885]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[0.0576, 1.6525, -1.3089]])
    )
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-SiglipForImageClassification"}
             )

    assert %Bumblebee.Vision.SigLipVision{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0952, 0.8452]])
    )
  end
end
