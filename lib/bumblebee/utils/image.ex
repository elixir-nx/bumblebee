defmodule Bumblebee.Utils.Image do
  @moduledoc false

  @compile {:no_warn_undefined, StbImage}

  @doc """
  Converts the given term to a batch of images.
  """
  def to_batched_tensor(%Nx.Tensor{} = image) do
    case Nx.rank(image) do
      3 ->
        Nx.new_axis(image, 0, :batch)

      4 ->
        image

      rank ->
        raise ArgumentError,
              "expected image to be a rank-3 image or a rank-4 batch, got rank: #{rank}"
    end
  end

  def to_batched_tensor(image) when is_struct(image, StbImage) do
    image
    |> StbImage.to_nx()
    |> Nx.new_axis(0, :batch)
  end

  @doc """
  Normalizes an image size to a `{height, width}` tuple.

  Accepts either an existing tuple or a single number used for both
  dimensions.
  """
  def normalize_size(size)

  def normalize_size({height, width}), do: {height, width}
  def normalize_size(size) when is_integer(size), do: {size, size}
end
