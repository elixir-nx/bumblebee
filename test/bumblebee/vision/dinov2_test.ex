defmodule Bumblebee.Vision.DinoV2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @tag :skip
  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "facebook/dinov2-base"})

    assert %Bumblebee.Vision.DinoV2{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs, debug: true)

    assert Nx.shape(outputs.pooled_state) == {1, 768}
    assert Nx.shape(outputs.hidden_state) == {1, 257, 768}

    # assert_all_close(
    #   outputs.hidden_state[[.., 1..3, 1..3]],
    #   Nx.tensor([
    #     [[-0.2075, 2.7865, 0.2361], [-0.3014, 2.5312, -0.6127], [-0.3460, 2.8741, 0.1988]]
    #   ]),
    #   atol: 1.0e-4
    # )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-6.0413055419921875, 2.106351137161255, -0.10576345771551132]]),
      atol: 1.0e-4
    )
  end

  @tag :skip
  test ":backbone" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "facebook/dinov2-base"}, architecture: :backbone)

    assert %Bumblebee.Vision.DinoV2{architecture: :backbone} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs, debug: true)

    assert Nx.shape(outputs.feature_maps) == {1, 16, 16, 768}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.2075, 2.7865, 0.2361], [-0.3014, 2.5312, -0.6127], [-0.3460, 2.8741, 0.1988]]
      ]),
      atol: 1.0e-4
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.0244, -0.0515, -0.1584]]),
      atol: 1.0e-4
    )
  end

  @tag :skip
  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "facebook/dinov2-small-imagenet1k-1-layer"})

    assert %Bumblebee.Vision.DinoV2{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.1596, 0.1818]]),
      atol: 1.0e-4
    )
  end
end
