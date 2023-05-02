defmodule Bumblebee.Text.LlamaTest do
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

    test "sequence classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model(
                 {:hf, "HuggingFaceH4/tiny-random-LlamaForSequenceClassification"}
               )

      assert %Bumblebee.Text.Llama{architecture: :for_sequence_classification} = spec
      input_ids = Nx.tensor([[1, 15043, 3186, 825, 29915, 29879, 701]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 1}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[-0.0977]]),
        atol: 1.0e-4
      )
    end

    test "causal language model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "seanmor5/tiny-llama-test"},
                 architecture: :for_causal_language_modeling
               )

      assert %Bumblebee.Text.Llama{architecture: :for_causal_language_modeling} = spec

      input_ids = Nx.tensor([[1, 15043, 3186, 825, 29915, 29879, 701]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 7, 32000}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [[0.0592, 0.1188, -0.1214], [-0.0331, 0.0335, -0.1808], [-0.1825, -0.0711, 0.0497]]
        ]),
        atol: 1.0e-2
      )
    end
  end
end
