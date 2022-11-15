defmodule Bumblebee.Vision.VitTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "google/vit-base-patch16-224"}, architecture: :base)

      assert %Bumblebee.Vision.Vit{architecture: :base} = spec

      inputs = %{"pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})}
      outputs = Axon.predict(model, params, inputs)

      # Pre-trained checkpoints by default do not use
      # the pooler layers
      assert Nx.shape(outputs.hidden_state) == {1, 197, 768}

      assert_all_close(
        outputs.hidden_state[[0, 0, 0..2]],
        Nx.tensor([0.4435, 0.4302, -0.1585]),
        atol: 1.0e-4
      )
    end

    test "image classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "google/vit-base-patch16-224"})

      assert %Bumblebee.Vision.Vit{architecture: :for_image_classification} = spec

      inputs = %{"pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})}
      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 1000}

      assert_all_close(
        outputs.logits[[0, 0..2]],
        Nx.tensor([0.0112, -0.5065, -0.7792]),
        atol: 1.0e-4
      )
    end

    test "masked image modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "google/vit-base-patch16-224-in21k"},
                 architecture: :for_masked_image_modeling
               )

      assert %Bumblebee.Vision.Vit{architecture: :for_masked_image_modeling} = spec

      # There is no pre-trained version on Hugging Face, so we use a fixed parameter
      params =
        update_in(params["decoder.0"]["kernel"], fn x ->
          # We use iota in the order of the pytorch kernel
          x
          |> Nx.transpose(axes: [3, 2, 1, 0])
          |> Nx.shape()
          |> Nx.iota(type: :f32)
          |> Nx.divide(Nx.size(x))
          |> Nx.transpose(axes: [2, 3, 1, 0])
        end)

      inputs = %{"pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})}
      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 224, 224, 3}

      assert_all_close(
        to_channels_first(outputs.logits)[[0, 0, 0..2, 0..2]],
        Nx.tensor([
          [-0.0103, -0.0275, -0.0447],
          [-0.2853, -0.3025, -0.3197],
          [-0.5603, -0.5774, -0.5946]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
