defmodule Bumblebee.Vision.ConvNextFeaturizer do
  @moduledoc """
  ConvNeXT featurizer for image data.

  ## Configuration

    * `:do_resize` - whether to resize (and optionally center crop)
      the input to the given `:size`. Defaults to `true`

    * `:size` - the size to resize the input to. If 384 or larger,
      the image is resized to (`:size`, `:size`). Otherwise, the
      shorter edge of the image is matched to `:size` / `:crop_pct`,
      then image is cropped to `:size`. Only has an effect if
      `:do_resize` is `true`. Defaults to `224`

    * `:resample` - the resizing method, either of `:nearest`, `:bilinear`,
      `:bicubic`, `:lanczos3`, `:lanczos5`. Defaults to `:bicubic`

    * `:crop_pct` - the percentage of the image to crop. Only has
      an effect if `:do_resize` is `true` and `:size` < 384. Defaults
      to `224 / 256`

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
            size: 224,
            resample: :bicubic,
            crop_pct: 224 / 256,
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

        cond do
          not config.do_resize ->
            images

          config.size >= 384 ->
            Image.resize(images, size: {config.size, config.size}, method: config.resample)

          true ->
            scale_size = floor(config.size / config.crop_pct)

            images
            |> Image.resize_short(size: scale_size, method: config.resample)
            |> Image.center_crop(size: {config.size, config.size})
        end
      end
      |> Nx.concatenate()

    images = Image.to_continuous(images, 0, 1)

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
