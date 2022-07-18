defmodule Bumblebee.Utils.Nx do
  @moduledoc false

  @doc """
  Maps the given container with the given function.

  When a tensor is given, the function is applied to this tensor.
  """
  def map(container_or_tensor, fun)

  def map(%Nx.Tensor{} = tensor, fun) do
    fun.(tensor)
  end

  def map(container, fun) do
    container
    |> Nx.Container.traverse(nil, &map(&1, fun))
    |> elem(0)
  end
end
