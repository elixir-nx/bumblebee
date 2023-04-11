defmodule Bumblebee.Utils.Image do
  @moduledoc false

  import Nx.Defn

  @doc """
  Converts the given term to a batch of image.
  """
  defn to_batched_tensor(image) do
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

  @doc """
  Normalizes an image size to a `{height, width}` tuple.

  Accepts either an existing tuple or a single number used for both
  dimensions.
  """
  def normalize_size(size)

  def normalize_size({height, width}), do: {height, width}
  def normalize_size(size) when is_integer(size), do: {size, size}

  @doc """
  Matches image against the desired number of channels and applies
  automatic conversions if applicable.
  """
  def normalize_channels(input, channels) do
    channel_axis = Nx.axis_index(input, -1)

    case {Nx.axis_size(input, channel_axis), channels} do
      {channels, channels} ->
        input

      {4, 3} ->
        Nx.slice_along_axis(input, 0, 3, axis: channel_axis)

      {1, 3} ->
        shape = input |> Nx.shape() |> put_elem(channel_axis, 3)
        Nx.broadcast(input, shape)

      {actual, expected} ->
        raise ArgumentError,
              "expected image with #{expected} channels, but got #{actual} and no automatic conversion applies"
    end
  end
end
