defmodule Bumblebee.Multimodal.BlipTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BlipForConditionalGeneration"}
             )

    assert %Bumblebee.Multimodal.Blip{architecture: :for_conditional_generation} = spec

    inputs = %{
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 1124}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1215, 0.0226, -0.1134], [0.1472, 0.1118, 0.1031], [-0.0687, 0.0104, 0.1781]]
      ])
    )
  end
end
