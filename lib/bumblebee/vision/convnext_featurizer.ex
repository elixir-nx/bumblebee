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

    * `:crop_pct` - the percentage of the image to crop. Only has
      an effect if `:do_resize` is `:true` and `:size` < 384. Defaults
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

  @behaviour Bumblebee.Featurizer

  defstruct do_resize: true,
            size: 224,
            # TODO: add support for configurable resampling
            # resample: :cubic,
            crop_pct: 224 / 256,
            do_normalize: true,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225]

  @impl true
  def config(config, opts \\ []) do
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def apply(config, images) when is_list(images) do
    images =
      images
      |> Enum.map(fn %StbImage{} = img ->
        cond do
          not config.do_resize ->
            to_tensor(img)

          config.size >= 384 ->
            img
            |> StbImage.resize(config.size, config.size)
            |> to_tensor()

          true ->
            scale_size = floor(config.size / config.crop_pct)

            img
            |> resize_short(scale_size)
            |> to_tensor()
            |> center_crop(config.size, config.size)
        end
      end)
      |> Nx.stack(name: :batch)

    if config.do_normalize do
      normalize(images, config.image_mean, config.image_std)
    else
      images
    end
  end

  # Scales the image such that the short edge matches `size`
  defp resize_short(%StbImage{} = img, size) when is_integer(size) do
    {height, width, _channels} = img.shape

    {short, long} = if height < width, do: {height, width}, else: {width, height}

    out_short = size
    out_long = floor(size * long / short)

    {out_height, out_width} =
      if height < width, do: {out_short, out_long}, else: {out_long, out_short}

    StbImage.resize(img, out_height, out_width)
  end

  defp to_tensor(%StbImage{} = img) do
    img
    |> StbImage.to_nx()
    |> Nx.transpose(axes: [:channels, :height, :width])
    |> Nx.divide(255.0)
  end

  defp center_crop(%Nx.Tensor{} = img, out_height, out_width) do
    {_channels, height, width} = Nx.shape(img)

    top = div(height - out_height, 2)
    bottom = top + out_height
    left = div(width - out_width, 2)
    right = left + out_width

    pad_config = [{0, 0, 0}, {-top, bottom - height, 0}, {-left, right - width, 0}]

    Nx.pad(img, 0, pad_config)
  end

  defp normalize(%Nx.Tensor{} = images, mean, std) do
    type = Nx.type(images)
    mean = mean |> Nx.tensor(type: type) |> Nx.reshape({1, :auto, 1, 1})
    std = std |> Nx.tensor(type: type) |> Nx.reshape({1, :auto, 1, 1})
    images |> Nx.subtract(mean) |> Nx.divide(std)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      Shared.data_into_config(data, config)
    end
  end
end
