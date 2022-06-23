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

    * `:resample` - the resizing method, either of `:nearest`, `:linear`,
      `:cubic`, `:lanczos3`, `:lanczos5`. Defaults to `:cubic`

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
  alias Bumblebee.Utils.Image

  @behaviour Bumblebee.Featurizer

  @compile {:no_warn_undefined, StbImage}

  defstruct do_resize: true,
            size: 224,
            resample: :cubic,
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

    images = Nx.divide(images, 255.0)

    if config.do_normalize do
      normalize(images, config.image_mean, config.image_std)
    else
      images
    end
  end

  defp normalize(images, mean, std) do
    type = Nx.type(images)
    mean = mean |> Nx.tensor(type: type) |> Nx.reshape({1, :auto, 1, 1})
    std = std |> Nx.tensor(type: type) |> Nx.reshape({1, :auto, 1, 1})
    images |> Nx.subtract(mean) |> Nx.divide(std)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      {resample, data} = Map.pop(data, "resample")

      config = Shared.data_into_config(data, config)

      load_resample(config, resample)
    end

    defp load_resample(config, resample) do
      resample =
        case resample do
          0 -> :nearest
          1 -> :lanczos3
          2 -> :linear
          3 -> :cubic
          _ -> nil
        end

      if resample, do: %{config | resample: resample}, else: config
    end
  end
end
