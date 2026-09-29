defmodule Bumblebee.Vision.Qwen3VLFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize images via the smart-resize algorithm"
    ],
    resize_method: [
      default: :bicubic,
      doc:
        "the resizing method, either of `:nearest`, `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`"
    ],
    normalize: [
      default: true,
      doc: "whether or not to normalize the input with mean and standard deviation"
    ],
    image_mean: [
      default: [0.5, 0.5, 0.5],
      doc: "the sequence of mean values for each channel, to be used when normalizing images"
    ],
    image_std: [
      default: [0.5, 0.5, 0.5],
      doc:
        "the sequence of standard deviations for each channel, to be used when normalizing images"
    ],
    patch_size: [
      default: 16,
      doc: "the spatial patch size"
    ],
    temporal_patch_size: [
      default: 2,
      doc: "the temporal patch size for video frames"
    ],
    merge_size: [
      default: 2,
      doc: "the merge factor for spatial patches"
    ],
    quality: [
      default: :medium,
      doc: """
      preset controlling the `:min_pixels` / `:max_pixels` caps used by smart-resize.
      One of `:low` (~256 visual tokens), `:medium` (~1280), or `:high` (16384).
      Ignored if `:min_pixels` and `:max_pixels` are both set explicitly.
      """
    ],
    min_pixels: [
      default: nil,
      doc: """
      explicit minimum total pixels after smart-resize. Overrides the `:quality`
      preset when set.
      """
    ],
    max_pixels: [
      default: nil,
      doc: """
      explicit maximum total pixels after smart-resize. Overrides the `:quality`
      preset when set.
      """
    ],
    max_patches: [
      default: nil,
      doc: """
      when set, pads `pixel_values` along the patches axis to this size with
      zeros. Required for compile-once-and-pad serving of variable-size
      images. Must be a multiple of `merge_size ** 2`.
      """
    ],
    max_num_images: [
      default: nil,
      doc: """
      when set, pads `image_grid_thw` to this many rows with `[0, 0, 0]`.
      Required alongside `:max_patches` for compile-once-and-pad serving.
      """
    ]
  ]

  @moduledoc """
  Qwen3-VL featurizer for image and video data.

  Accepts a single image, a list of images, or a `%{video: [frame, ...]}`
  map. When given multiple images they are concatenated into a single
  flat sequence of patches; per-image grid dimensions are returned as
  `image_grid_thw`.

  ## Quality profiles

  Smart-resize caps the total number of pixels passed through the
  patchifier. The `:quality` preset is a convenience over the explicit
  `:min_pixels` / `:max_pixels` keys:

    * `:low` — ~256 visual tokens per image (fastest, lowest detail)
    * `:medium` — ~1280 visual tokens per image (default)
    * `:high` — up to 16384 visual tokens per image (full Qwen ceiling)

  Set `:min_pixels` and/or `:max_pixels` to override the preset.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Featurizer
  @behaviour Bumblebee.Configurable

  alias Bumblebee.Utils.Image

  @impl true
  def config(featurizer, opts) do
    Shared.put_config_attrs(featurizer, opts)
  end

  @impl true
  def process_input(featurizer, input) do
    factor = featurizer.patch_size * featurizer.merge_size
    {min_pixels, max_pixels} = resolve_pixel_bounds(featurizer, factor)

    per_image =
      for image_or_video <- normalize_input(input) do
        process_one(featurizer, image_or_video, min_pixels, max_pixels, factor)
      end

    pixel_values =
      per_image
      |> Enum.map(& &1.pixel_values)
      |> Nx.concatenate(axis: 0)

    image_grid_thw =
      per_image
      |> Enum.map(& &1.grid_thw)
      |> Nx.stack()

    {pixel_values, image_grid_thw} =
      maybe_pad_to_max(pixel_values, image_grid_thw, featurizer)

    %{
      "pixel_values" => pixel_values,
      "image_grid_thw" => image_grid_thw
    }
  end

  defp maybe_pad_to_max(pixel_values, image_grid_thw, featurizer) do
    pixel_values = maybe_pad_patches(pixel_values, featurizer)
    image_grid_thw = maybe_pad_grid_thw(image_grid_thw, featurizer)
    {pixel_values, image_grid_thw}
  end

  defp maybe_pad_patches(pixel_values, %{max_patches: nil}), do: pixel_values

  defp maybe_pad_patches(pixel_values, featurizer) do
    {num_patches, flat} = Nx.shape(pixel_values)
    max_patches = featurizer.max_patches
    merge_sq = featurizer.merge_size * featurizer.merge_size

    unless rem(max_patches, merge_sq) == 0 do
      raise ArgumentError,
            ":max_patches (#{max_patches}) must be a multiple of merge_size**2 " <>
              "(= #{merge_sq})"
    end

    if num_patches > max_patches do
      raise ArgumentError,
            "featurizer produced #{num_patches} patches but :max_patches is " <>
              "#{max_patches}; raise :max_patches or lower :quality / :max_pixels"
    end

    pad_rows = max_patches - num_patches

    if pad_rows == 0 do
      pixel_values
    else
      padding = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(pixel_values)), {pad_rows, flat})
      Nx.concatenate([pixel_values, padding], axis: 0)
    end
  end

  defp maybe_pad_grid_thw(image_grid_thw, %{max_num_images: nil}), do: image_grid_thw

  defp maybe_pad_grid_thw(image_grid_thw, featurizer) do
    {num_images, 3} = Nx.shape(image_grid_thw)
    max_num_images = featurizer.max_num_images

    if num_images > max_num_images do
      raise ArgumentError,
            "got #{num_images} images but :max_num_images is #{max_num_images}"
    end

    pad_rows = max_num_images - num_images

    if pad_rows == 0 do
      image_grid_thw
    else
      padding = Nx.broadcast(Nx.tensor(0, type: Nx.type(image_grid_thw)), {pad_rows, 3})
      Nx.concatenate([image_grid_thw, padding], axis: 0)
    end
  end

  defp normalize_input(input) when is_list(input), do: input
  defp normalize_input(%{image: _} = input), do: [input]
  defp normalize_input(%{video: _} = input), do: [input]
  defp normalize_input(input), do: [%{image: input}]

  defp process_one(featurizer, %{video: frames}, min_pixels, max_pixels, factor)
       when is_list(frames) do
    process_frames(featurizer, frames, min_pixels, max_pixels, factor)
  end

  defp process_one(featurizer, %{image: image}, min_pixels, max_pixels, factor) do
    process_frames(featurizer, [image], min_pixels, max_pixels, factor)
  end

  defp process_one(featurizer, image, min_pixels, max_pixels, factor) do
    process_frames(featurizer, [image], min_pixels, max_pixels, factor)
  end

  defp process_frames(featurizer, frames, min_pixels, max_pixels, factor) do
    num_channels = length(featurizer.image_mean)

    batched_frames =
      Enum.map(frames, fn frame ->
        frame
        |> Image.to_batched_tensor()
        |> Nx.as_type(:f32)
        |> Image.normalize_channels(num_channels)
      end)

    [first | _] = batched_frames
    {1, height, width, _} = Nx.shape(first)

    {target_h, target_w} =
      if featurizer.resize do
        smart_resize(height, width, min_pixels, max_pixels, factor)
      else
        h = max(factor, round_to_multiple(height, factor))
        w = max(factor, round_to_multiple(width, factor))
        {h, w}
      end

    mean = Nx.tensor(featurizer.image_mean)
    std = Nx.tensor(featurizer.image_std)

    processed_frames =
      Enum.map(batched_frames, fn frame ->
        frame
        |> NxImage.resize({target_h, target_w}, method: featurizer.resize_method)
        |> NxImage.to_continuous(0, 1)
        |> maybe_normalize(featurizer, mean, std)
        |> Nx.squeeze(axes: [0])
      end)

    stacked = Nx.stack(processed_frames)
    {stacked, temporal} = ensure_temporal(stacked, featurizer.temporal_patch_size)

    patches_t = div(temporal, featurizer.temporal_patch_size)
    patches_h = div(target_h, featurizer.patch_size)
    patches_w = div(target_w, featurizer.patch_size)

    pixel_values = window_patchify(stacked, featurizer, patches_t, patches_h, patches_w)

    %{
      pixel_values: pixel_values,
      grid_thw: Nx.tensor([patches_t, patches_h, patches_w], type: :s64)
    }
  end

  defp maybe_normalize(images, %{normalize: false}, _mean, _std), do: images
  defp maybe_normalize(images, _, mean, std), do: NxImage.normalize(images, mean, std)

  defp ensure_temporal(stacked, temporal_patch_size) do
    {temporal, _, _, _} = Nx.shape(stacked)

    target =
      if temporal < temporal_patch_size do
        temporal_patch_size
      else
        div(temporal, temporal_patch_size) * temporal_patch_size
      end

    cond do
      target == temporal ->
        {stacked, temporal}

      target > temporal ->
        last = stacked[(temporal - 1)..(temporal - 1)//1]
        pad = Nx.tile(last, [target - temporal, 1, 1, 1])
        {Nx.concatenate([stacked, pad], axis: 0), target}

      target < temporal ->
        {Nx.slice_along_axis(stacked, 0, target, axis: 0), target}
    end
  end

  # Arranges patches in "windowed" order so that every group of
  # merge_size * merge_size consecutive patches forms a contiguous
  # spatial merge block. This lets the vision encoder's patch merger
  # reshape {N, hidden} -> {N/merge^2, merge^2 * hidden} without
  # needing to know per-image grid dimensions.
  defp window_patchify(stacked, featurizer, patches_t, patches_h, patches_w) do
    {_temporal, _height, _width, channels} = Nx.shape(stacked)
    patch_size = featurizer.patch_size
    temporal_patch_size = featurizer.temporal_patch_size
    merge_size = featurizer.merge_size
    merged_h = div(patches_h, merge_size)
    merged_w = div(patches_w, merge_size)

    stacked
    |> Nx.reshape({
      patches_t,
      temporal_patch_size,
      merged_h,
      merge_size,
      patch_size,
      merged_w,
      merge_size,
      patch_size,
      channels
    })
    |> Nx.transpose(axes: [0, 2, 5, 3, 6, 8, 1, 4, 7])
    |> Nx.reshape({
      patches_t * merged_h * merged_w * merge_size * merge_size,
      channels * temporal_patch_size * patch_size * patch_size
    })
  end

  defp smart_resize(height, width, min_pixels, max_pixels, factor) do
    ratio = max(height, width) / min(height, width)

    if ratio > 200 do
      raise ArgumentError,
            "image aspect ratio is #{Float.round(ratio, 2)}, " <>
              "which exceeds the supported limit of 200"
    end

    h_bar = max(factor, round_to_multiple(height, factor))
    w_bar = max(factor, round_to_multiple(width, factor))

    cond do
      h_bar * w_bar > max_pixels ->
        beta = :math.sqrt(height * width / max_pixels)
        h2 = floor_to_multiple(height / beta, factor)
        w2 = floor_to_multiple(width / beta, factor)
        {max(factor, h2), max(factor, w2)}

      h_bar * w_bar < min_pixels ->
        beta = :math.sqrt(min_pixels / (height * width))
        h2 = ceil_to_multiple(height * beta, factor)
        w2 = ceil_to_multiple(width * beta, factor)
        {h2, w2}

      true ->
        {h_bar, w_bar}
    end
  end

  defp round_to_multiple(value, factor) do
    round(value / factor) * factor
  end

  defp floor_to_multiple(value, factor) do
    trunc(value / factor) * factor
  end

  defp ceil_to_multiple(value, factor) do
    trunc(Float.ceil(value / factor)) * factor
  end

  defp resolve_pixel_bounds(featurizer, factor) do
    f2 = factor * factor

    {default_min, default_max} =
      case featurizer.quality do
        :low ->
          {4 * f2, 256 * f2}

        :medium ->
          {4 * f2, 1280 * f2}

        :high ->
          {4 * f2, 16384 * f2}

        other ->
          raise ArgumentError,
                "invalid :quality #{inspect(other)}, expected :low, :medium, or :high"
      end

    min_pixels = featurizer.min_pixels || default_min
    max_pixels = featurizer.max_pixels || default_max

    if min_pixels > max_pixels do
      raise ArgumentError,
            "min_pixels (#{min_pixels}) must not exceed max_pixels (#{max_pixels})"
    end

    {min_pixels, max_pixels}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(featurizer, data) do
      import Shared.Converters

      opts =
        convert!(data,
          resize: {"do_resize", boolean()},
          resize_method: {"resample", resize_method()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())},
          patch_size: {"patch_size", number()},
          temporal_patch_size: {"temporal_patch_size", number()},
          merge_size: {"merge_size", number()},
          min_pixels: {"min_pixels", number()},
          max_pixels: {"max_pixels", number()}
        )

      @for.config(featurizer, opts)
    end
  end
end
