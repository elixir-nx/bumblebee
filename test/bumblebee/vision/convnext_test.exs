defmodule Bumblebee.Vision.ConvNextTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-ConvNextModel"})

    assert %Bumblebee.Vision.ConvNext{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 7, 7, 40}
    assert Nx.shape(outputs.pooled_state) == {1, 40}

    assert_all_close(
      outputs.hidden_state[[.., 1..2, 1..2, 1..2]],
      Nx.tensor([[[[0.3924, -0.2330], [0.3924, -0.2330]], [[0.3924, -0.2330], [0.3924, -0.2330]]]])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[2.2793, -1.3236, -1.0714]]),
      atol: 1.0e-3
    )
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-ConvNextForImageClassification"}
             )

    assert %Bumblebee.Vision.ConvNext{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[0.0047, -0.1457]])
    )
  end
end
