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

  defp nodes(axon, ids, nodes) do
    if MapSet.member?(ids, axon.id) do
      {ids, nodes}
    else
      {ids, nodes} = do_nodes(axon, ids, nodes)
      {MapSet.put(ids, axon.id), [axon | nodes]}
    end
  end

  defp do_nodes(%Axon{op: :container, parent: [container]}, ids, nodes) do
    Nx.Container.reduce(container, {ids, nodes}, &nodes_reducer/2)
  end

  defp do_nodes(%Axon{parent: parent}, ids, nodes) do
    Enum.reduce(parent || [], {ids, nodes}, &nodes_reducer/2)
  end

  defp nodes_reducer(%Axon{} = axon, {ids, nodes}) do
    nodes(axon, ids, nodes)
  end

  defp nodes_reducer(container, {ids, nodes}) do
    Nx.Container.reduce(container, {ids, nodes}, &nodes_reducer/2)
  end

  @doc """
  Extracts the underlying value from Axon container.
  """
  @spec unwrap_container(Axon.t()) :: term()
  def unwrap_container(%Axon{op: :container, parent: [container]}) do
    container
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

  @doc """
  Adds the given prefix to all layer names.

  Note that input nodes are kept intact.
  """
  @spec prefix_names(Axon.t(), String.t()) :: Axon.t()
  def prefix_names(%Axon{} = model, prefix) do
    Axon.map_nodes(model, fn
      %{op: :input} = node ->
        node

      node ->
        update_in(node.name, fn name ->
          fn op, op_counts ->
            prefix <> name.(op, op_counts)
          end
        end)
    end)
  end

  @doc """
  Replaces input nodes with nodes given in the input map.
  """
  @spec plug_inputs(Axon.t(), %{String.t() => Axon.t()}) :: Axon.t()
  def plug_inputs(%Axon{} = model, inputs) do
    replace_nodes(model, fn
      %{op: :input} = node ->
        name = node.name.(:input, %{})
        inputs[name] || Bumblebee.Layers.none()

      _node ->
        :keep
    end)
  end

  @doc """
  Optionally replaces nodes in the given graph.

  `fun` is applied exactly once to each node. The node is replaced
  completely, including the node parents.
  """
  @spec replace_nodes(Axon.t(), (Axon.t() -> Axon.t() | :keep)) :: Axon.t()
  def replace_nodes(%Axon{} = axon, fun) when is_function(fun, 1) do
    {axon, _cache} = do_replace_nodes(axon, %{}, fun)
    axon
  end

  defp do_replace_nodes(%Axon{id: id, parent: parents} = node, cache, fun) do
    case cache do
      %{^id => result} ->
        {result, cache}

      %{} ->
        {result, cache} =
          case fun.(node) do
            :keep ->
              {parents, cache} = deep_map_reduce(parents, cache, &do_replace_nodes(&1, &2, fun))
              {%{node | parent: parents}, cache}

            %Axon{} = new_node ->
              {new_node, cache}

            other ->
              raise ArgumentError,
                    "function passed to map_nodes must return an" <>
                      " Axon struct, got #{inspect(other)}"
          end

        {result, Map.put(cache, id, result)}
    end
  end

  defp deep_map_reduce(%Axon{} = node, acc, fun) do
    fun.(node, acc)
  end

  defp deep_map_reduce(nodes, acc, fun) when is_list(nodes) do
    Enum.map_reduce(nodes, acc, &deep_map_reduce(&1, &2, fun))
  end

  defp deep_map_reduce(container, acc, fun) do
    Nx.Container.traverse(container, acc, &deep_map_reduce(&1, &2, fun))
  end
end
