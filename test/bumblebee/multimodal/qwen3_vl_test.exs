defmodule Bumblebee.Multimodal.Qwen3VLTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    # Tiny model created with /tmp/create_tiny_qwen3vl_v4.py (transformers 4.57.3):
    # - text_config: vocab_size=1024, hidden_size=64, num_hidden_layers=2,
    #                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    #                intermediate_size=128
    # - vision_config: depth=2, hidden_size=32, num_heads=4, intermediate_size=64,
    #                  out_hidden_size=64, patch_size=14, spatial_merge_size=2,
    #                  temporal_patch_size=2
    #
    # Reference values from /tmp/generate_reference_v2.py (seed=0):
    # model = Qwen3VLForConditionalGeneration.from_pretrained(model_path)
    # outputs = model(input_ids=torch.tensor([[10, 20, 30, 40, 50, 60, 0, 0]]),
    #                 attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]))
    # outputs.logits[0, 0:3, 0:5].numpy()

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "roulis/tiny-random-Qwen3VLForConditionalGeneration"})

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
          [0.0410, 0.0745, -0.0977, 0.0099, 0.2705],
          [-0.0504, 0.1776, -0.0481, -0.0269, 0.1630],
          [-0.1887, 0.0889, -0.1113, -0.1756, 0.0805]
        ]
      ]),
      atol: 1.0e-4
    )
  end

  test "vision pathway runs end-to-end with image_grid_thw" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "roulis/tiny-random-Qwen3VLForConditionalGeneration"})

    factor = spec.vision_spec.patch_size * spec.vision_spec.spatial_merge_size

    featurizer =
      Bumblebee.configure(Bumblebee.Vision.Qwen3VLFeaturizer,
        patch_size: spec.vision_spec.patch_size,
        merge_size: spec.vision_spec.spatial_merge_size,
        temporal_patch_size: spec.vision_spec.temporal_patch_size,
        min_pixels: 4 * factor * factor,
        max_pixels: 64 * factor * factor
      )

    image = Nx.iota({64, 64, 3}, type: :u8)
    image_inputs = Bumblebee.apply_featurizer(featurizer, image)

    [grid_t, grid_h, grid_w] = Nx.to_flat_list(image_inputs["image_grid_thw"])
    merge_size = spec.vision_spec.spatial_merge_size
    visual_tokens = grid_t * div(grid_h, merge_size) * div(grid_w, merge_size)

    image_token_id = spec.image_token_id
    input_ids = List.duplicate(image_token_id, visual_tokens) ++ [1, 2, 3]
    attention_mask = List.duplicate(1, length(input_ids))

    inputs = %{
      "input_ids" => Nx.tensor([input_ids]),
      "attention_mask" => Nx.tensor([attention_mask]),
      "pixel_values" => image_inputs["pixel_values"],
      "image_grid_thw" => image_inputs["image_grid_thw"]
    }

    outputs = Axon.predict(model, params, inputs)

    expected_seq = visual_tokens + 3
    assert {1, ^expected_seq, 1024} = Nx.shape(outputs.logits)
  end

  test "vision pathway accepts multiple images of different sizes" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "roulis/tiny-random-Qwen3VLForConditionalGeneration"})

    factor = spec.vision_spec.patch_size * spec.vision_spec.spatial_merge_size

    featurizer =
      Bumblebee.configure(Bumblebee.Vision.Qwen3VLFeaturizer,
        patch_size: spec.vision_spec.patch_size,
        merge_size: spec.vision_spec.spatial_merge_size,
        temporal_patch_size: spec.vision_spec.temporal_patch_size,
        min_pixels: 4 * factor * factor,
        max_pixels: 64 * factor * factor
      )

    images = [Nx.iota({56, 56, 3}, type: :u8), Nx.iota({84, 56, 3}, type: :u8)]
    image_inputs = Bumblebee.apply_featurizer(featurizer, images)

    assert {2, 3} = Nx.shape(image_inputs["image_grid_thw"])

    merge_size = spec.vision_spec.spatial_merge_size

    visual_tokens =
      image_inputs["image_grid_thw"]
      |> Nx.to_batched(1)
      |> Enum.map(fn row ->
        [t, h, w] = Nx.to_flat_list(row)
        t * div(h, merge_size) * div(w, merge_size)
      end)
      |> Enum.sum()

    image_token_id = spec.image_token_id
    input_ids = List.duplicate(image_token_id, visual_tokens) ++ [1, 2]
    attention_mask = List.duplicate(1, length(input_ids))

    inputs = %{
      "input_ids" => Nx.tensor([input_ids]),
      "attention_mask" => Nx.tensor([attention_mask]),
      "pixel_values" => image_inputs["pixel_values"],
      "image_grid_thw" => image_inputs["image_grid_thw"]
    }

    outputs = Axon.predict(model, params, inputs)

    expected_seq = visual_tokens + 2
    assert {1, ^expected_seq, 1024} = Nx.shape(outputs.logits)
  end
end
