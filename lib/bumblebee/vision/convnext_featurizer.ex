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

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Featurizer
  @behaviour Bumblebee.Configurable

  alias Bumblebee.Utils.Image

  @impl true
  def config(featurizer, opts) do
    Shared.put_config_attrs(featurizer, opts)
  end

  @impl true
  def process_input(featurizer, images) do
    images = List.wrap(images)

    for image <- images do
      images =
        image
        |> Image.to_batched_tensor()
        |> Nx.as_type(:f32)
        |> Image.normalize_channels(length(featurizer.image_mean))

      cond do
        not featurizer.resize ->
          images

        featurizer.size >= 384 ->
          NxImage.resize(images, {featurizer.size, featurizer.size},
            method: featurizer.resize_method
          )

        true ->
          scale_size = floor(featurizer.size / featurizer.crop_percentage)

          images
          |> NxImage.resize_short(scale_size, method: featurizer.resize_method)
          |> NxImage.center_crop({featurizer.size, featurizer.size})
      end
    end
    |> Nx.concatenate()
  end

  @impl true
  def batch_template(featurizer, batch_size) do
    num_channels = length(featurizer.image_mean)
    Nx.template({batch_size, featurizer.size, featurizer.size, num_channels}, :f32)
  end

  @impl true
  def process_batch(featurizer, images) do
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

    %{"pixel_values" => images}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(featurizer, data) do
      import Shared.Converters

      opts =
        convert!(data,
          resize: {"do_resize", boolean()},
          size: {"size", size()},
          resize_method: {"resample", resize_method()},
          crop_percentage: {"crop_pct", number()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())}
        )

      @for.config(featurizer, opts)
    end

    defp size() do
      # Note that in contrast to other featurizers, in this case size
      # is always a single number and its meaning depends on the input
      # size. huggingface/transformers put it under the "shortest_edge"
      # key, but we keep it as a single number as it is more clear.
      fn name, value ->
        case value do
          %{"shortest_edge" => size} ->
            {:ok, size}

          size when is_number(size) ->
            {:ok, size}

          _ ->
            {:error,
             "expected #{inspect(name)} to be a number or a map with shortest_edge, got: #{inspect(value)}"}
        end
      end
    end
  end
end
