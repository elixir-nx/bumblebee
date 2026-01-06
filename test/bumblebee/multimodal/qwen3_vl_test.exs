defmodule Bumblebee.Multimodal.Qwen3VLTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @tag :skip
  test ":for_conditional_generation" do
    # TODO: Create tiny-random checkpoint at bumblebee-testing/tiny-random-Qwen3VLForConditionalGeneration
    # and get reference values from Python
    #
    # The tiny model was created with:
    # - text_config: vocab_size=1024, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    #                num_key_value_heads=2, head_dim=16, intermediate_size=128
    # - vision_config: depth=2, embed_dim=32, num_heads=4, mlp_ratio=2, patch_size=8,
    #                  temporal_patch_size=2, spatial_merge_size=2, hidden_size=64
    #
    # Reference values obtained from Python (transformers 4.57.3):
    # torch.manual_seed(42)
    # outputs = model(input_ids=torch.tensor([[10, 20, 30, 40, 50, 60, 0, 0]]),
    #                 attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]))
    # outputs.logits[:, 0:3, 0:5].numpy()

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Qwen3VLForConditionalGeneration"}
             )

    assert %Bumblebee.Multimodal.Qwen3VL{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 1024}

    # Reference values from Python (transformers 4.57.3)
    assert_all_close(
      outputs.logits[[.., 0..2, 0..4]],
      Nx.tensor([
        [
          [-0.01338646, -0.01154798, 0.01520334, 0.09433511, -0.20700514],
          [0.02179704, -0.12912436, 0.15642744, -0.0126619, -0.309812],
          [0.01208664, 0.0299146, -0.12953377, -0.03512848, -0.05375983]
        ]
      ]),
      atol: 1.0e-4
    )
  end
end
