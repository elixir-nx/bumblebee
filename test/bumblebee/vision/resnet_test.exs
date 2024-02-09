defmodule Bumblebee.Vision.ResNetTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-ResNetModel"})

    assert %Bumblebee.Vision.ResNet{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 7, 7, 40}
    assert Nx.shape(outputs.pooled_state) == {1, 40}

    assert_all_close(
      to_channels_first(outputs.hidden_state)[[.., 2..3, 2..3, 2..3]],
      Nx.tensor([[[[0.0000, 0.0000], [0.0000, 0.0000]], [[0.9835, 0.9835], [0.9835, 0.9835]]]]),
      atol: 1.0e-4
    )

    assert_all_close(Nx.sum(outputs.hidden_state), Nx.tensor(209.6328), atol: 1.0e-4)

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[0.0275, 0.0095, 0.8921]]),
      atol: 1.0e-4
    )

    assert_all_close(Nx.sum(outputs.pooled_state), Nx.tensor(4.2782), atol: 1.0e-4)
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-ResNetForImageClassification"}
             )

    assert %Bumblebee.Vision.ResNet{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 3}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.1053, 0.2160, -0.0331]]),
      atol: 1.0e-4
    )
  end
end
