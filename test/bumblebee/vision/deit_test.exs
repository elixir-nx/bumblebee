defmodule Bumblebee.Vision.DeitTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "facebook/deit-base-distilled-patch16-224"},
                 architecture: :base
               )

      assert %Bumblebee.Vision.Deit{architecture: :base} = spec

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, %{"pixel_values" => input})

      # Pre-trained checkpoints by default do not use
      # the pooler layers
      assert Nx.shape(output.last_hidden_state) == {1, 198, 768}

      assert_all_close(
        output.last_hidden_state[[0, 0, 0..2]],
        Nx.tensor([-0.0738, -0.2792, -0.0235]),
        atol: 1.0e-4
      )
    end

    test "image classification model with teacher" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "facebook/deit-base-distilled-patch16-224"})

      assert %Bumblebee.Vision.Deit{architecture: :for_image_classification_with_teacher} = spec

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, %{"pixel_values" => input})

      assert Nx.shape(output.logits) == {1, 1000}

      assert_all_close(
        output.logits[[0, 0..2]],
        Nx.tensor([-0.7490, 0.7397, 0.6383]),
        atol: 1.0e-4
      )
    end

    test "masked image modeling model" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "facebook/deit-base-distilled-patch16-224"},
                 architecture: :for_masked_image_modeling
               )

      assert %Bumblebee.Vision.Deit{architecture: :for_masked_image_modeling} = spec

      # There is no pre-trained version on Hugging Face, so we use a fixed parameter
      params =
        update_in(params["decoder.0"]["kernel"], fn x ->
          x |> Nx.iota(type: :f32) |> Nx.divide(Nx.size(x))
        end)

      input = %{"pixel_values" => Nx.broadcast(0.5, {1, 3, 224, 224})}
      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 3, 224, 224}

      assert_all_close(
        output.logits[[0, 0, 0..2, 0..2]],
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
