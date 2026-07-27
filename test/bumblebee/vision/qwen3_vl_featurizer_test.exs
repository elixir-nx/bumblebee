defmodule Bumblebee.Vision.Qwen3VLFeaturizerTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Vision.Qwen3VLFeaturizer

  defp synthetic_image(height, width, channels \\ 3) do
    Nx.iota({height, width, channels}, type: :u8)
    |> Nx.remainder(255)
  end

  defp featurizer(opts \\ []) do
    defaults = [
      patch_size: 16,
      temporal_patch_size: 2,
      merge_size: 2
    ]

    Bumblebee.configure(Qwen3VLFeaturizer, Keyword.merge(defaults, opts))
  end

  test "produces pixel_values and image_grid_thw for a single image" do
    image = synthetic_image(64, 64)
    inputs = Bumblebee.apply_featurizer(featurizer(), image)

    # 4x4 = 16 patches; flat = channels * temporal_patch * patch * patch = 3*2*16*16 = 1536
    assert {16, 1536} = Nx.shape(inputs["pixel_values"])
    assert {1, 3} = Nx.shape(inputs["image_grid_thw"])

    # 64x64 image, patch=16 -> 4x4 patches, temporal duplicated 1->2 -> patches_t=1
    assert Nx.to_flat_list(inputs["image_grid_thw"]) == [1, 4, 4]
  end

  test "smart_resize preserves aspect ratio and rounds to factor multiples" do
    # 96x64 input. factor = 16 * 2 = 32. 96 = 3*32, 64 = 2*32 — already aligned.
    image = synthetic_image(96, 64)
    inputs = Bumblebee.apply_featurizer(featurizer(), image)

    [_t, grid_h, grid_w] = Nx.to_flat_list(inputs["image_grid_thw"])
    # patch_size=16: 96/16=6, 64/16=4
    assert grid_h == 6
    assert grid_w == 4

    expected_patches = grid_h * grid_w
    assert {^expected_patches, _} = Nx.shape(inputs["pixel_values"])
  end

  test "max_pixels caps the resized image" do
    # 1024x1024 with max_pixels=256 visual tokens forces a strong downscale.
    image = synthetic_image(1024, 1024)
    factor = 32
    max_pixels = 256 * factor * factor

    inputs =
      Bumblebee.apply_featurizer(
        featurizer(min_pixels: 4 * factor * factor, max_pixels: max_pixels),
        image
      )

    [_t, grid_h, grid_w] = Nx.to_flat_list(inputs["image_grid_thw"])
    merge_size = 2
    visual_tokens = div(grid_h, merge_size) * div(grid_w, merge_size)

    assert visual_tokens <= 256
  end

  test ":low quality produces fewer visual tokens than :high" do
    image = synthetic_image(2048, 1536)

    [_t, low_h, low_w] =
      Bumblebee.apply_featurizer(featurizer(quality: :low), image)["image_grid_thw"]
      |> Nx.to_flat_list()

    [_t, high_h, high_w] =
      Bumblebee.apply_featurizer(featurizer(quality: :high), image)["image_grid_thw"]
      |> Nx.to_flat_list()

    assert low_h * low_w < high_h * high_w
  end

  test "supports multiple images of different sizes in one call" do
    images = [synthetic_image(64, 64), synthetic_image(96, 64)]
    inputs = Bumblebee.apply_featurizer(featurizer(), images)

    assert {2, 3} = Nx.shape(inputs["image_grid_thw"])
    assert Nx.to_flat_list(inputs["image_grid_thw"]) == [1, 4, 4, 1, 6, 4]

    # Total patches = 4*4 + 6*4 = 40; flat = 3*2*16*16 = 1536
    assert {40, 1536} = Nx.shape(inputs["pixel_values"])
  end

  test "windowed layout: every 4 consecutive patches form one 2x2 merge block" do
    # A 64x64 image gives a 4x4 patch grid. With merge_size=2 there are
    # 2x2 = 4 merge blocks of 4 patches each. Patches inside one block
    # come from one spatial region of the resized image, so their flat
    # patch features must be pairwise close. We verify the layout by
    # checking that within each block-of-4 the variance is much smaller
    # than the variance across blocks.
    image =
      Nx.iota({64, 64, 3}, type: :f32)
      |> Nx.divide(64 * 64 * 3)

    inputs = Bumblebee.apply_featurizer(featurizer(normalize: false), image)

    grouped = Nx.reshape(inputs["pixel_values"], {4, 4, 1536})
    within_block_var = grouped |> Nx.variance(axes: [1]) |> Nx.mean() |> Nx.to_number()

    across_block_var =
      grouped
      |> Nx.mean(axes: [1])
      |> Nx.variance(axes: [0])
      |> Nx.mean()
      |> Nx.to_number()

    assert within_block_var < across_block_var
  end

  test "raises on extreme aspect ratios" do
    image = synthetic_image(1, 400)

    assert_raise ArgumentError, ~r/aspect ratio/, fn ->
      Bumblebee.apply_featurizer(featurizer(), image)
    end
  end

  test "raises when min_pixels exceeds max_pixels" do
    image = synthetic_image(64, 64)

    assert_raise ArgumentError, ~r/min_pixels/, fn ->
      Bumblebee.apply_featurizer(featurizer(min_pixels: 10_000, max_pixels: 1_000), image)
    end
  end

  test "pads pixel_values to :max_patches with zeros" do
    image = synthetic_image(64, 64)
    inputs = Bumblebee.apply_featurizer(featurizer(max_patches: 64), image)

    assert {64, 1536} = Nx.shape(inputs["pixel_values"])
    # First 16 patches are real, rest are zero-padded
    real_block = inputs["pixel_values"][[0..15, ..]]
    pad_block = inputs["pixel_values"][[16..63, ..]]
    assert Nx.to_number(Nx.sum(Nx.abs(pad_block))) == 0.0
    refute Nx.to_number(Nx.sum(Nx.abs(real_block))) == 0.0
  end

  test "pads image_grid_thw with [0, 0, 0] rows" do
    image = synthetic_image(64, 64)
    inputs = Bumblebee.apply_featurizer(featurizer(max_num_images: 3), image)

    assert {3, 3} = Nx.shape(inputs["image_grid_thw"])
    assert Nx.to_flat_list(inputs["image_grid_thw"]) == [1, 4, 4, 0, 0, 0, 0, 0, 0]
  end

  test "raises when :max_patches is not a multiple of merge_size**2" do
    image = synthetic_image(64, 64)

    assert_raise ArgumentError, ~r/multiple of merge_size/, fn ->
      Bumblebee.apply_featurizer(featurizer(max_patches: 17), image)
    end
  end

  test "raises when image needs more patches than :max_patches" do
    image = synthetic_image(96, 96)

    assert_raise ArgumentError, ~r/raise :max_patches/, fn ->
      Bumblebee.apply_featurizer(featurizer(max_patches: 16), image)
    end
  end
end
