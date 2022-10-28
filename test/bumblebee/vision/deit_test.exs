defmodule Bumblebee.Vision.DeitTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/deit-base-distilled-patch16-224"},
                 architecture: :base
               )

      assert %Bumblebee.Vision.Deit{architecture: :base} = spec

      inputs = %{"pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})}
      outputs = Axon.predict(model, params, inputs)

      # Pre-trained checkpoints by default do not use
      # the pooler layers
      assert Nx.shape(outputs.hidden_state) == {1, 198, 768}

      assert_all_close(
        outputs.hidden_state[[0, 0, 0..2]],
        Nx.tensor([-0.0738, -0.2792, -0.0235]),
        atol: 1.0e-4
      )
    end

    test "image classification model with teacher" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/deit-base-distilled-patch16-224"})

      assert %Bumblebee.Vision.Deit{architecture: :for_image_classification_with_teacher} = spec

      inputs = %{"pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})}
      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 1000}

      assert_all_close(
        outputs.logits[[0, 0..2]],
        Nx.tensor([-0.7490, 0.7397, 0.6383]),
        atol: 1.0e-4
      )
    end

    test "masked image modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/deit-base-distilled-patch16-224"},
                 architecture: :for_masked_image_modeling
               )

      assert %Bumblebee.Vision.Deit{architecture: :for_masked_image_modeling} = spec

      # There is no pre-trained version on Hugging Face, so we use a fixed parameter
      params =
        update_in(params["decoder.0"]["kernel"], fn x ->
          # We use iota in the order of the pytorch kernel
          x
          |> Nx.transpose(axes: [3, 2, 1, 0])
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
          [-0.0159, 0.0084, 0.0326],
          [0.3719, 0.3961, 0.4204],
          [0.7597, 0.7839, 0.8082]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
