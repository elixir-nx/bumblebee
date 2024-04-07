defmodule Bumblebee.Vision.BitFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize the input to the given `:size`"
    ],
    size: [
      default: %{shortest_edge: 448},
      doc: """
      the size to resize the input to, either `%{height: ..., width: ...}` or `%{shortest_edge: ...}`.
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
      default: %{height: 448, width: 448},
      doc: """
      the size to center crop the image to, given as `%{height: ..., width: ...}`. Only has an effect
      if `:center_crop` is `true`
      """
    ],
    rescale: [
      default: true,
      doc: "whether to rescale the input by the given `:rescale_factor`"
    ],
    rescale_factor: [
      default: 0.00392156862745098,
      doc: """
      the factor by which to rescale the input. A single number
      Only has an effect if `:rescale` is `true`
      """
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
    ]
  ]

  @moduledoc """
  BiT featurizer for image data.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Featurizer
  @behaviour Bumblebee.Configurable

  alias Bumblebee.Utils.Image

  @impl true
  def config(featurizer, opts) do
    featurizer = Shared.put_config_attrs(featurizer, opts)

    if featurizer.resize and Shared.featurizer_size_fixed?(featurizer.size) and
         not featurizer.center_crop do
      raise ArgumentError,
            "the resize shape depends on the input shape and cropping is disabled." <>
              "You must either configure a fixed size or enable cropping"
    end

    featurizer
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

      images =
        if featurizer.resize do
          size = Shared.featurizer_resize_size(images, featurizer.size)
          NxImage.resize(images, size, method: featurizer.resize_method)
        else
          images
        end

      if featurizer.center_crop do
        %{height: height, width: width} = featurizer.crop_size
        NxImage.center_crop(images, {height, width})
      else
        images
      end
    end
    |> Nx.concatenate()
  end

  @impl true
  def batch_template(featurizer, batch_size) do
    num_channels = length(featurizer.image_mean)

    {height, width} =
      case featurizer do
        %{center_crop: true, crop_size: %{height: height, width: width}} ->
          {height, width}

        %{resize: true, size: %{height: height, width: width}} ->
          {height, width}
      end

    Nx.template({batch_size, height, width, num_channels}, :f32)
  end

  @impl true
  def process_batch(featurizer, images) do
    images =
      if featurizer.rescale do
        Nx.multiply(images, featurizer.rescale_factor)
      else
        images
      end

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
          size: {"size", image_size(single_as: :shortest_edge)},
          resize_method: {"resample", resize_method()},
          center_crop: {"do_center_crop", boolean()},
          crop_size: {"crop_size", image_size()},
          rescale: {"do_rescale", boolean()},
          rescale_factor: {"rescale_factor", number()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())}
        )

      @for.config(featurizer, opts)
    end
  end
end
