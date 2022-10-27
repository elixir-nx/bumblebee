defmodule Bumblebee.Text.ClipTextTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"},
                 module: Bumblebee.Text.ClipText,
                 architecture: :base
               )

      assert %Bumblebee.Text.ClipText{architecture: :base} = spec

      inputs = %{
        "input_ids" =>
          Nx.tensor([
            [49406, 320, 1125, 539, 320, 2368, 49407],
            [49406, 320, 1125, 539, 320, 1929, 49407]
          ]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {2, 7, 512}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.5844, 0.3685, -2.0744], [-0.9600, 1.0018, -0.2415], [-0.5957, -0.1719, 0.4689]],
          [[-0.5844, 0.3685, -2.0744], [-0.0025, 0.1219, -0.0435], [0.0661, 0.1142, 0.0056]]
        ]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.pooled_state[[0..-1//1, 1..3]],
        Nx.tensor([[0.1658, 0.8876, 10.6313], [0.0130, 0.1167, 0.0371]]),
        atol: 1.0e-4
      )
    end
  end
end
