defmodule Bumblebee.Text.CamembertTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "camembert-base"}, architecture: :base)

      assert %Bumblebee.Text.Roberta{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 402, 232, 328, 740, 1140, 12695, 69, 1588, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 10, 768}

      assert_all_close(
        outputs.hidden_state[[.., 0..2, 0..2]],
        Nx.tensor([
          [[0.0592, 0.0688, 0.0185], [0.0024, 0.1443, -0.1943], [0.0102, 0.2724, -0.2474]]
        ]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "camembert-base"})

      assert %Bumblebee.Text.Roberta{architecture: :for_masked_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 402, 232, 328, 740, 1140, 12695, 69, 1588, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 10, 32005}

      assert_all_close(
        outputs.logits[[.., 0..2, 0..2]],
        Nx.tensor([
          [[18.4213, -4.5504, 6.5444], [-1.2791, -2.4822, 2.8339], [-2.5561, -4.3118, -1.3791]]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
