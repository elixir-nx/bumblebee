defmodule Bumblebee.Utils.Axon do
  @moduledoc false

  @doc """
  Returns a list with all nodes in the given model and their names.
  """
  @spec nodes_with_names(Axon.t()) :: list({Axon.t(), String.t()})
  def nodes_with_names(model) do
    model
    |> nodes()
    |> Enum.map_reduce(%{}, fn axon, op_counts ->
      name = axon.name.(axon.op, op_counts)
      op_counts = Map.update(op_counts, axon.op, 1, &(&1 + 1))
      {{axon, name}, op_counts}
    end)
    |> elem(0)
  end

  defp nodes(axon) do
    {_ids, nodes} = nodes(axon, MapSet.new(), [])
    Enum.reverse(nodes)
  end

  defp nodes(%Axon{op: :container, parent: [container]} = axon, ids, nodes) do
    {ids, nodes} = Nx.Container.reduce(container, {ids, nodes}, &nodes_reducer/2)
    maybe_add_node(axon, ids, nodes)
  end

  defp nodes(%Axon{parent: parent} = axon, ids, nodes) do
    {ids, nodes} = Enum.reduce(parent || [], {ids, nodes}, &nodes_reducer/2)
    maybe_add_node(axon, ids, nodes)
  end

  defp nodes_reducer(%Axon{} = axon, {ids, nodes}) do
    nodes(axon, ids, nodes)
  end

  defp nodes_reducer(container, {ids, nodes}) do
    Nx.Container.reduce(container, {ids, nodes}, &nodes_reducer/2)
  end

  defp maybe_add_node(axon, ids, nodes) do
    if MapSet.member?(ids, axon.id) do
      {ids, nodes}
    else
      {MapSet.put(ids, axon.id), [axon | nodes]}
    end
  end

  @doc """
  Extracts the underlying value from Axon container.
  """
  @spec unwrap_container(Axon.t()) :: term()
  def unwrap_container(%Axon{op: :container, parent: [container]}) do
    container
  end

  @doc """
  Runs initializer for the given parameter.
  """
  @spec init_param(Axon.t(), %Axon.Parameter{}) :: Nx.tensor()
  def init_param(layer, param) do
    dtype = layer.policy.params

    case param.shape do
      {:tuple, shapes} ->
        shapes
        |> Enum.map(fn shape ->
          apply(Axon.Initializers, param.initializer, [[type: dtype, shape: shape]])
        end)
        |> List.to_tuple()

      shape ->
        apply(Axon.Initializers, param.initializer, [[type: dtype, shape: shape]])
    end
  end
end
