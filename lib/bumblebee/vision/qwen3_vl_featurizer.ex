defmodule Bumblebee.Vision.Qwen3VLFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize the input to the given `:size`"
    ],
    size: [
      default: %{height: 448, width: 448},
      doc: """
      the size to resize the input to, given as `%{height: ..., width: ...}`. Only has
      an effect if `:resize` is `true`
      """
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
    ]
  ]

  @moduledoc """
  Qwen3-VL featurizer for image and video data.

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
    images = normalize_input(input)

    for image_or_video <- images do
      process_single_input(featurizer, image_or_video)
    end
    |> Nx.concatenate()
  end

  defp normalize_input(input) when is_list(input), do: input
  defp normalize_input(%{image: _} = input), do: [input]
  defp normalize_input(%{video: _} = input), do: [input]
  defp normalize_input(input), do: [%{image: input}]

  defp process_single_input(featurizer, %{video: frames}) when is_list(frames) do
    # Video input: process multiple frames
    frames
    |> Enum.map(&process_frame(featurizer, &1))
    |> Nx.stack()
    # Stack frames along temporal dimension: {batch, temporal, height, width, channels}
    |> Nx.transpose(axes: [1, 0, 2, 3, 4])
  end

  defp process_single_input(featurizer, %{image: image}) do
    # Single image: temporal dimension = 1
    image
    |> process_frame(featurizer)
    |> Nx.new_axis(1)

    # Shape: {batch, 1, height, width, channels}
  end

  defp process_single_input(featurizer, image) do
    # Assume it's just an image
    process_single_input(featurizer, %{image: image})
  end

  defp process_frame(frame, featurizer) do
    frame =
      frame
      |> Image.to_batched_tensor()
      |> Nx.as_type(:f32)
      |> Image.normalize_channels(length(featurizer.image_mean))

    # Qwen3VL requires image dimensions to be divisible by patch_size * merge_size
    factor = featurizer.patch_size * featurizer.merge_size

    {_, h, w, _} = Nx.shape(frame)

    # Compute target size - round to nearest multiple of factor
    target_h = round_to_multiple(h, factor)
    target_w = round_to_multiple(w, factor)

    # Ensure minimum size
    target_h = max(target_h, factor)
    target_w = max(target_w, factor)

    NxImage.resize(frame, {target_h, target_w}, method: featurizer.resize_method)
  end

  defp round_to_multiple(value, factor) do
    div(value + div(factor, 2), factor) * factor
  end

  @impl true
  def batch_template(featurizer, batch_size) do
    # Get height/width from size config, defaulting to 224 if not specified
    {height, width} =
      case featurizer.size do
        %{height: h, width: w} -> {h, w}
        %{shortest_edge: edge} when edge < 10000 -> {edge, edge}
        _ -> {224, 224}
      end

    num_channels = length(featurizer.image_mean)
    # Output shape includes temporal dimension: {batch, channels, temporal, height, width}
    # For template, we use temporal=1 (single image case)
    %{
      "pixel_values" => Nx.template({batch_size, num_channels, 1, height, width}, :f32)
    }
  end

  @impl true
  def process_batch(featurizer, images) do
    # images shape: {batch, temporal, height, width, channels}
    images = NxImage.to_continuous(images, 0, 1)

    images =
      if featurizer.normalize do
        NxImage.normalize(
          images,
          Nx.tensor(featurizer.image_mean),
          Nx.tensor(featurizer.image_std)
        )
      else
        images
      end

    # Convert to {batch, channels, temporal, height, width} for model
    images = Nx.transpose(images, axes: [0, 4, 1, 2, 3])

    %{"pixel_values" => images}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(featurizer, data) do
      import Shared.Converters

      opts =
        convert!(data,
          resize: {"do_resize", boolean()},
          size: {"size", image_size()},
          resize_method: {"resample", resize_method()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())},
          patch_size: {"patch_size", number()},
          temporal_patch_size: {"temporal_patch_size", number()},
          merge_size: {"merge_size", number()}
        )

      @for.config(featurizer, opts)
    end
  end
end
