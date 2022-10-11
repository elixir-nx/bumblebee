defmodule Bumblebee.Vision.DeitFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize (and optionally center crop) the input to the given `:size`"
    ],
    size: [
      default: 256,
      doc: """
      the size to resize the input to. Either a single number or a `{height, width}` tuple.
      Only has an effect if `:resize` is `true`
      """
    ],
    resize_method: [
      default: :bicubic,
      doc:
        "the resizing method, either of `:nearest`, `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`"
    ],
    center_crop: [
      default: true,
      doc: """
      whether to crop the input at the center. If the input size is smaller than `:crop_size` along
      any edge, the image is padded with zeros and then center cropped
      """
    ],
    crop_size: [
      default: 224,
      doc: "the size to center crop the image to. Only has an effect if `:center_crop` is `true`"
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
  DeiT featurizer for image data.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  alias Bumblebee.Utils.Image

  @behaviour Bumblebee.Featurizer
  @behaviour Bumblebee.Configurable

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

        if config.resize do
          size = Image.normalize_size(config.size)
          Image.resize(images, size: size, method: config.resize_method)
        else
          images
        end
      end
      |> Nx.concatenate()

    images = Image.to_continuous(images, 0, 1)

    images =
      if config.center_crop do
        Image.center_crop(images, size: {config.crop_size, config.crop_size})
      else
        images
      end

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
          size: {"size", one_of([number(), tuple([number(), number()])])},
          resize_method: {"resample", resize_method()},
          center_crop: {"do_center_crop", boolean()},
          crop_size: {"crop_size", number()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())}
        )

      @for.config(config, opts)
    end
  end
end
