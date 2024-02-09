defmodule Bumblebee.Vision.DinoV2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

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

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[-6.0413055419921875, 2.106351137161255, -0.10576345771551132]]),
      atol: 1.0e-4
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
      Nx.tensor([-3.0963878631591797, -1.9401777982711792, -1.4899224042892456]),
      atol: 1.0e-4
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
      Nx.tensor([-3.083641290664673, -1.8309074640274048, -0.28923606872558594]),
      atol: 1.0e-4
    )
  end
end
