defmodule Bumblebee.Vision.VitTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :capture_log
    @tag :slow
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "google/vit-base-patch16-224"}, architecture: :base)

      assert %Bumblebee.Vision.Vit{architecture: :base} = config

      input = %{"pixel_values" => Nx.broadcast(0.5, {1, 3, 224, 224})}
      output = Axon.predict(model, params, input)

      # Pre-trained checkpoints by default do not use
      # the pooler layers
      assert Nx.shape(output.last_hidden_state) == {1, 197, 768}

      assert_all_close(
        output.last_hidden_state[[0, 0, 0..2]],
        Nx.tensor([0.4435, 0.4302, -0.1585]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    test "image classification model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "google/vit-base-patch16-224"})

      assert %Bumblebee.Vision.Vit{architecture: :for_image_classification} = config

      input = %{"pixel_values" => Nx.broadcast(0.5, {1, 3, 224, 224})}
      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 1000}

      assert_all_close(
        output.logits[[0, 0..2]],
        Nx.tensor([0.0112, -0.5065, -0.7792]),
        atol: 1.0e-4
      )
    end
  end
end
