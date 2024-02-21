defmodule Bumblebee.Vision.DinoV2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Dinov2Model"})

    assert %Bumblebee.Vision.DinoV2{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 226, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-1.1210, -0.3567, -0.4570], [-1.0003, -0.8821, -0.5325], [-0.6919, -0.5334, -0.4568]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.7099, -0.6118, 0.7679]])
    )
  end

  test ":base with position embedding interpolation (different input size)" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Dinov2Model"})

    assert %Bumblebee.Vision.DinoV2{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 64, 64, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 1025, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    # Note: the interpolation has a slightly different implementation
    # in PyTorch, so the numbers don't match exactly, but this is used
    # at inference and should be fine in practice. We do a low-precision
    # comparison for the reference.

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-1.2287, -0.2291, -0.4323], [-1.1548, -0.4430, -0.4710], [-1.0547, -0.7580, -0.4654]]
      ]),
      atol: 1.0e-1
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.7270, -0.5913, 0.7701]]),
      atol: 1.0e-2
    )
  end

  test ":base with swiglu ffn" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Dinov2Model-use_swiglu_ffn-True"}
             )

    assert %Bumblebee.Vision.DinoV2{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 226, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-1.4022, 0.2361, 0.6539], [-1.0799, -0.3041, 0.3125], [-0.7367, -0.0650, 0.6671]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-0.5637, 0.7523, 1.0458]])
    )
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-Dinov2ForImageClassification"}
             )

    assert %Bumblebee.Vision.DinoV2{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[0.1091, 0.0126]])
    )
  end

  test ":backbone" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Dinov2Backbone"})

    assert %Bumblebee.Vision.DinoV2{architecture: :backbone} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {feature_map} = outputs.feature_maps

    assert Nx.shape(feature_map) == {1, 15, 15, 32}

    assert_all_close(
      feature_map[[.., 1..2, 1..2, 1..2]],
      Nx.tensor([[[[1.3373, 0.6393], [0.5469, 1.4045]], [[1.1879, 0.7435], [1.1777, 0.6638]]]])
    )
  end

  test ":backbone with different feature map subset" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Dinov2Backbone"},
               spec_overrides: [backbone_output_indices: [0, 2]]
             )

    assert %Bumblebee.Vision.DinoV2{architecture: :backbone} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {feature_map0, feature_map2} = outputs.feature_maps

    assert Nx.shape(feature_map0) == {1, 15, 15, 32}
    assert Nx.shape(feature_map2) == {1, 15, 15, 32}

    assert_all_close(
      feature_map0[[.., 1..2, 1..2, 1..2]],
      Nx.tensor([[[[0.8425, 0.5487], [0.2003, 1.2553]], [[0.6486, 0.4550], [1.3376, 0.7091]]]])
    )

    assert_all_close(
      feature_map2[[.., 1..2, 1..2, 1..2]],
      Nx.tensor([[[[1.3373, 0.6393], [0.5469, 1.4045]], [[1.1879, 0.7435], [1.1777, 0.6638]]]])
    )
  end
end
