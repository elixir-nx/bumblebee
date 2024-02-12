defmodule Bumblebee.Vision.VitTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-ViTModel"})

    assert %Bumblebee.Vision.Vit{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 226, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.2075, 2.7865, 0.2361], [-0.3014, 2.5312, -0.6127], [-0.3460, 2.8741, 0.1988]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.0244, -0.0515, -0.1584]])
    )
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-ViTForImageClassification"}
             )

    assert %Bumblebee.Vision.Vit{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.1596, 0.1818]])
    )
  end

  test ":for_masked_image_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-ViTForMaskedImageModeling"}
             )

    assert %Bumblebee.Vision.Vit{architecture: :for_masked_image_modeling} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.pixel_values) == {1, 30, 30, 3}

    assert_all_close(
      to_channels_first(outputs.pixel_values)[[.., 1..2, 1..2, 1..2]],
      Nx.tensor([[[[0.0752, -0.0192], [-0.0252, 0.0232]], [[0.0548, -0.0216], [0.0728, -0.1687]]]])
    )
  end
end
