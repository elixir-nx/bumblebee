defmodule Bumblebee.Multimodal.BlipTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "conditional generation model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Salesforce/blip-image-captioning-base"})

      assert %Bumblebee.Multimodal.Blip{architecture: :for_conditional_generation} = spec

      inputs = %{
        "decoder_input_ids" =>
          Nx.tensor([
            [101, 2019],
            [101, 2019]
          ]),
        "decoder_attention_mask" => Nx.tensor([[1, 1], [1, 1]]),
        "pixel_values" =>
          Nx.concatenate([
            Nx.broadcast(0.25, {1, 384, 384, 3}),
            Nx.broadcast(0.75, {1, 384, 384, 3})
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {2, 2, 30524}

      assert_all_close(
        outputs.logits[[.., .., 1..3]],
        Nx.tensor([
          [[-3.6837, -3.6838, -3.6837], [-1.4808, -1.4809, -1.4808]],
          [[-3.5190, -3.5191, -3.5190], [-1.4715, -1.4715, -1.4715]]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
