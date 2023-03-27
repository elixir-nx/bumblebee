defmodule Bumblebee.Vision.BlipVisionTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Salesforce/blip-image-captioning-base"},
                 module: Bumblebee.Vision.BlipVision,
                 architecture: :base
               )

      assert %Bumblebee.Vision.BlipVision{architecture: :base} = spec

      inputs = %{
        "pixel_values" => Nx.broadcast(0.5, {1, 384, 384, 3})
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 577, 768}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.5337, 1.1098, 0.4768], [-0.7984, 0.9996, -0.2640], [-0.1782, 0.8242, 0.4417]]
        ]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.pooled_state[[0..-1//1, 1..3]],
        Nx.tensor([[-0.0882, -0.3926, -0.5420]]),
        atol: 1.0e-4
      )
    end
  end
end
