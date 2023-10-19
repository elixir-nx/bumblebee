defmodule Bumblebee.Text.MistralTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "seanmor5/tiny-llama-test"}, architecture: :base)

      assert %Bumblebee.Text.Llama{architecture: :base} = spec

      input_ids = Nx.tensor([[1, 15043, 3186, 825, 29915, 29879, 701]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 7, 32}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-0.4411, -1.9037, 0.9454], [0.8148, -1.4606, 0.0076], [0.9480, 0.6038, 0.1649]]
        ]),
        atol: 1.0e-2
      )
    end
  end
end