defmodule Bumblebee.Utils.Axon do
  @moduledoc false

  @doc """
  Returns a list with all nodes in the given model and their names.
  """
  @spec nodes_with_names(Axon.t()) :: list({Axon.t(), String.t()})
  def nodes_with_names(%Axon{nodes: nodes}) do
    nodes
    |> Map.values()
    |> Enum.map_reduce(%{}, fn axon, op_counts ->
      name = axon.name.(axon.op, op_counts)
      op_counts = Map.update(op_counts, axon.op, 1, &(&1 + 1))
      {{axon, name}, op_counts}
    end)
    |> elem(0)
  end

  @doc """
  Recursively zips matching `Nx.Container`s by applying `fun` to
  corresponding `Axon` nodes in these containers.
  """
  @spec container_zip_with(
          Nx.Container.t(),
          Nx.Container.t(),
          (Axon.t(), Axon.t() -> Axon.t())
        ) :: Nx.Container.t()
  def container_zip_with(left, right, fun) do
    case Nx.Container.traverse(left, leaves(right), &do_zip_with(&1, &2, fun)) do
      {merged, []} ->
        merged

      {_merged, _leftover} ->
        raise ArgumentError, "cannot zip containers with incompatible structure"
    end
  end

  defp leaves(container) do
    container
    |> Nx.Container.reduce([], fn x, acc -> [x | acc] end)
    |> Enum.reverse()
  end

  defp do_zip_with(left, [right | right_leaves], fun) do
    case {left, right} do
      {%Axon{} = left, %Axon{} = right} ->
        {fun.(left, right), right_leaves}

      {left, right} ->
        {do_zip_with(left, right, fun), right_leaves}
    end
  end

  @doc """
  Recursively maps `Axon` nodes in the given `Nx.Container`.
  """
  @spec container_map(Nx.Container.t(), (Axon.t() -> Axon.t())) :: Nx.Container.t()
  def container_map(container, fun) do
    container
    |> Nx.Container.traverse(nil, fn
      %Axon{} = x, acc ->
        {fun.(x), acc}

      container, acc ->
        {container_map(container, fun), acc}
    end)
    |> elem(0)
  end
end
