defmodule Bumblebee.Vision.ConvNextFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize (and optionally center crop) the input to the given `:size`"
    ],
    size: [
      default: 224,
      doc: """
      the size to resize the input to. If 384 or larger, the image is resized to (`:size`, `:size`).
      Otherwise, the shorter edge of the image is matched to `:size` / `:crop_percentage`, then image
      is cropped to `:size`. Only has an effect if `:resize` is `true`
      """
    ],
    resize_method: [
      default: :bicubic,
      doc:
        "the resizing method, either of `:nearest`, `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`"
    ],
    crop_percentage: [
      default: 224 / 256,
      doc:
        "the percentage of the image to crop. Only has an effect if `:resize` is `true` and `:size` < 384"
    ],
    normalize: [
      default: true,
      doc: "whether or not to normalize the input with mean and standard deviation"
    ],
    image_mean: [
      default: [0.485, 0.456, 0.406],
      doc: "the sequence of mean values for each channel, to be used when normalizing images"
    ],
    image_std: [
      default: [0.229, 0.224, 0.225],
      doc:
        "the sequence of standard deviations for each channel, to be used when normalizing images"
    ]
  ]

  @moduledoc """
  ConvNeXT featurizer for image data.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  alias Bumblebee.Utils.Image

  @behaviour Bumblebee.Featurizer

  defstruct Shared.option_defaults(options)

  @impl true
  def config(config, opts \\ []) do
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def apply(config, images) do
    images = List.wrap(images)

    images =
      for image <- images do
        images = image |> Image.to_batched_tensor() |> Nx.as_type(:f32)

        cond do
          not config.resize ->
            images

          config.size >= 384 ->
            Image.resize(images, size: {config.size, config.size}, method: config.resize_method)

          true ->
            scale_size = floor(config.size / config.crop_percentage)

            images
            |> Image.resize_short(size: scale_size, method: config.resize_method)
            |> Image.center_crop(size: {config.size, config.size})
        end
      end
      |> Nx.concatenate()

    images = Image.to_continuous(images, 0, 1)

    images =
      if config.normalize do
        Image.normalize(images, Nx.tensor(config.image_mean), Nx.tensor(config.image_std))
      else
        images
      end

    %{"pixel_values" => images}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      import Shared.Converters

      opts =
        convert!(data,
          resize: {"do_resize", boolean()},
          size: {"size", number()},
          resize_method: {"resample", resize_method()},
          crop_percentage: {"crop_pct", number()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())}
        )

      @for.config(config, opts)
    end
  end
end
