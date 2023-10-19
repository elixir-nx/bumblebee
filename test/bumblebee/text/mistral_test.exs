defmodule Bumblebee.Text.MistralTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "echarlaix/tiny-random-mistral"}, architecture: :base)

      assert %Bumblebee.Text.Mistral{architecture: :base} = spec

      input_ids = Nx.tensor([[1, 6312, 28709, 1526, 28808]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 5, 32}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [
            [-1.1513, -0.3565, -1.3482],
            [0.5468, 0.5652, -0.4141],
            [-1.2177, -0.7919, -0.7064]
          ]
        ]),
        atol: 1.0e-2
      )
    end
  end
end
