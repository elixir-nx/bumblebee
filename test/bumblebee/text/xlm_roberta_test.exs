defmodule Bumblebee.Text.XlmRobertaTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "xlm-roberta-base"}, architecture: :base)

      assert %Bumblebee.Text.Roberta{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 581, 10323, 111, 9942, 83, 250_001, 6, 5, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 10, 768}

      assert_all_close(
        outputs.hidden_state[[.., 0..2, 0..2]],
        Nx.tensor([
          [[0.4921, 0.3050, 0.1307], [-0.0038, -0.0187, -0.0312], [0.0248, -0.0300, 0.0382]]
        ]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "xlm-roberta-base"})

      assert %Bumblebee.Text.Roberta{architecture: :for_masked_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 581, 10323, 111, 9942, 83, 250_001, 6, 5, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 10, 250_002}

      assert_all_close(
        outputs.logits[[.., 0..2, 0..2]],
        Nx.tensor([
          [[64.3345, 0.1994, 38.5827], [28.9445, -1.5083, 73.2020], [21.0732, -1.0673, 52.7042]]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
