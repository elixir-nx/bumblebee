defmodule Bumblebee.Utils.Image do
  @moduledoc false

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

  def to_batched_tensor(term) do
    tensor = Nx.Defn.jit_apply(&Function.identity/1, [term], compiler: Nx.Defn.Evaluator)
    to_batched_tensor(tensor)
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
