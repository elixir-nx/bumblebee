defmodule Bumblebee.Text.BlipTextTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Salesforce/blip-vqa-base"},
                 module: Bumblebee.Text.BlipText,
                 architecture: :base
               )

      assert %Bumblebee.Text.BlipText{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]]),
        "attention_mask" => Nx.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 11, 768}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.0219, 0.0386, -0.0164], [-0.0205, 0.0398, -0.0155], [-0.0242, 0.0405, -0.0186]]
        ]),
        atol: 1.0e-4
      )
    end

    test "causal language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Salesforce/blip-image-captioning-base"},
                 module: Bumblebee.Text.BlipText,
                 architecture: :for_causal_language_modeling
               )

      assert %Bumblebee.Text.BlipText{architecture: :for_causal_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]]),
        "attention_mask" => Nx.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 11, 30524}

      assert_all_close(
        outputs.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([[[0.0525, 0.0526, 0.0525], [0.0433, 0.0434, 0.0433], [0.0833, 0.0834, 0.0833]]]),
        atol: 1.0e-4
      )
    end
  end
end
