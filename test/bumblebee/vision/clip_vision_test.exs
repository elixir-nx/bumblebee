defmodule Bumblebee.Vision.ClipVisionTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"},
                 module: Bumblebee.Vision.ClipVision,
                 architecture: :base
               )

      assert %Bumblebee.Vision.ClipVision{architecture: :base} = spec

      inputs = %{
        "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 50, 768}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[0.3465, -0.3939, -0.5297], [0.3588, -0.2529, -0.5606], [0.3958, -0.2688, -0.5367]]
        ]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.pooled_state[[.., 1..3]],
        Nx.tensor([[0.3602, 0.3658, -0.2337]]),
        atol: 1.0e-4
      )
    end

    test "embedding model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"},
                 module: Bumblebee.Vision.ClipVision,
                 architecture: :for_embedding
               )

      assert %Bumblebee.Vision.ClipVision{architecture: :for_embedding} = spec

      inputs = %{
        "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.embedding) == {1, 512}

      assert_all_close(
        outputs.embedding[[.., 1..3]],
        Nx.tensor([[-0.3381, -0.0196, -0.4053]]),
        atol: 1.0e-4
      )
    end
  end
end
