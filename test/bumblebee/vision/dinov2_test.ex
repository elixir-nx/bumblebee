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

  test ":backbone" do
    {:ok, spec} = Bumblebee.load_spec({:hf, "facebook/dinov2-base"}, architecture: :backbone)

    spec =
      Bumblebee.configure(spec,
        stage_names: [
          "stem",
          "stage1",
          "stage2",
          "stage3",
          "stage4",
          "stage5",
          "stage6",
          "stage7",
          "stage8",
          "stage9",
          "stage10",
          "stage11",
          "stage12"
        ],
        output_features: ["stage12"]
      )

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "facebook/dinov2-base"},
               architecture: :backbone,
               spec: spec
             )

    assert %Bumblebee.Vision.DinoV2{architecture: :backbone} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs, debug: true)

    last_feature_map = Map.get(outputs.feature_maps, "stage12")
    assert Nx.shape(last_feature_map) == {1, 16, 16, 768}

    assert_all_close(
      last_feature_map[[0, 1, 1, 1..3]],
      Nx.tensor([-3.0963878631591797, -1.9401777982711792, -1.4899224042892456])
    )
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "facebook/dinov2-small-imagenet1k-1-layer"})

    assert %Bumblebee.Vision.DinoV2{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs, debug: true)

    assert Nx.shape(outputs.logits) == {1, 1, 1000}

    # prediction =
    #   Nx.argmax(outputs.logits, axis: -1) |> Nx.squeeze() |> Nx.to_number()
    # assert prediction == 281

    assert_all_close(
      Nx.squeeze(outputs.logits)[[0..2]],
      Nx.tensor([-3.083641290664673, -1.8309074640274048, -0.28923606872558594])
    )
  end
end
