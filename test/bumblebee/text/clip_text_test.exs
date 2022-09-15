defmodule Bumblebee.Text.ClipTextTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"},
                 module: Bumblebee.Text.ClipText,
                 architecture: :base
               )

      assert %Bumblebee.Text.ClipText{architecture: :base} = config

      input = %{
        "input_ids" =>
          Nx.tensor([
            [49406, 320, 1125, 539, 320, 2368, 49407],
            [49406, 320, 1125, 539, 320, 1929, 49407]
          ]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {2, 7, 512}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.5844, 0.3685, -2.0744], [-0.9600, 1.0018, -0.2415], [-0.5957, -0.1719, 0.4689]],
          [[-0.2595, 0.0761, -0.1841], [-0.1127, 0.8916, -1.3848], [0.0503, 1.1131, -1.6429]]
        ]),
        atol: 1.0e-4
      )

      assert_all_close(
        output.pooler_output[[0..-1//1, 1..3]],
        Nx.tensor([[0.1658, 0.8876, 10.6313], [0.0189, 1.2740, 10.2831]]),
        atol: 1.0e-4
      )
    end
  end
end