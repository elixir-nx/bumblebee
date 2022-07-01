defmodule Bumblebee.Vision.VitFeaturizer do
  @moduledoc """
  ViT featurizer for image data.

  ## Configuration

    * `:do_resize` - whether to resize the input to the given `:size`.
      Defaults to `true`

    * `:size` - the size to resize the input to. Either a single number
      or a `{height, width}` tuple. Only has an effect if `:do_resize`
      is `true`. Defaults to `224`

    * `:resample` - the resizing method, either of `:nearest`, `:linear`,
      `:cubic`, `:lanczos3`, `:lanczos5`. Defaults to `:linear`

    * `:do_normalize` - whether or not to normalize the input with
      mean and standard deviation. Defaults to `true`

    * `:image_mean` - the sequence of mean values for each channel,
      to be used when normalizing images. Defaults to `[0.5, 0.5, 0.5]`

    * `:image_std` - the sequence of standard deviations for each
      channel, to be used when normalizing images. Defaults to
      `[0.5, 0.5, 0.5]`

  """

  alias Bumblebee.Shared
  alias Bumblebee.Utils.Image

  @behaviour Bumblebee.Featurizer

  defstruct do_resize: true,
            size: 224,
            resample: :linear,
            do_normalize: true,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5]

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

    if config.do_normalize do
      Image.normalize(images, Nx.tensor(config.image_mean), Nx.tensor(config.image_std))
    else
      images
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_resample_method("resample")
      |> Shared.data_into_config(config)
    end
  end
end
