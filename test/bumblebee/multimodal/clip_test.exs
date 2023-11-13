defmodule Bumblebee.Multimodal.ClipTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"})

      assert %Bumblebee.Multimodal.Clip{architecture: :base} = spec

      inputs = %{
        "input_ids" =>
          Nx.tensor([
            [49_406, 320, 1125, 539, 320, 2368, 49_407],
            [49_406, 320, 1125, 539, 320, 1929, 49_407]
          ]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]),
        "pixel_values" =>
          Nx.concatenate([
            Nx.broadcast(0.25, {1, 224, 224, 3}),
            Nx.broadcast(0.75, {1, 224, 224, 3})
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits_per_text) == {2, 2}

      assert_all_close(
        outputs.logits_per_text,
        Nx.tensor([[22.7866, 22.8397], [23.2389, 22.8406]]),
        atol: 1.0e-4
      )

      assert Nx.shape(outputs.logits_per_image) == {2, 2}

      assert_all_close(
        outputs.logits_per_image,
        Nx.tensor([[22.7866, 23.2389], [22.8397, 22.8406]]),
        atol: 1.0e-4
      )
    end
  end
end
