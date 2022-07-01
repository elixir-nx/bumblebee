defmodule Bumblebee.Vision.DeitFeaturizer do
  @moduledoc """
  DeiT featurizer for image data.

  ## Configuration

    * `:do_resize` - whether to resize (and optionally center crop)
      the input to the given `:size`. Defaults to `true`

    * `:size` - the size to resize the input to. Either a single number
      or a `{height, width}` tuple. Only has an effect if `:do_resize`
      is `true`. Defaults to `256`

    * `:resample` - the resizing method, either of `:nearest`, `:linear`,
      `:cubic`, `:lanczos3`, `:lanczos5`. Defaults to `:cubic`

    * `:do_center_crop` - whether to crop the input at the center. If
      the input size is smaller than `:crop_size` along any edge, the
      image is padded with zeros and then center cropped. Defaults to
      `true`

    * `:crop_size` - the size to center crop the image to. Only has an
      effect if `:do_center_crop` is `true`. Defaults to `224`

    * `:do_normalize` - whether or not to normalize the input with
      mean and standard deviation. Defaults to `true`

    * `:image_mean` - the sequence of mean values for each channel,
      to be used when normalizing images. Defaults to `[0.485, 0.456, 0.406]`

    * `:image_std` - the sequence of standard deviations for each
      channel, to be used when normalizing images. Defaults to
      `[0.229, 0.224, 0.225]`

  """

  alias Bumblebee.Shared
  alias Bumblebee.Utils.Image

  @behaviour Bumblebee.Featurizer

  defstruct do_resize: true,
            size: 256,
            resample: :cubic,
            do_center_crop: true,
            crop_size: 224,
            do_normalize: true,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225]

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

        if config.do_resize do
          size = Image.normalize_size(config.size)
          Image.resize(images, size: size, method: config.resample)
        else
          images
        end
      end
      |> Nx.concatenate()

    images = Nx.divide(images, 255.0)

    images =
      if config.do_center_crop do
        Image.center_crop(images, size: {config.crop_size, config.crop_size})
      else
        images
      end

    images =
      if config.do_normalize do
        Image.normalize(images, Nx.tensor(config.image_mean), Nx.tensor(config.image_std))
      else
        images
      end

    %{"pixel_values" => images}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_resample_method("resample")
      |> Shared.data_into_config(config)
    end
  end
end
