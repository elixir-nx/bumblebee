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
        {container_zip_with(left, right, fun), right_leaves}
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
    {id_mapping, new_nodes} =
      Axon.reduce_nodes(axon, {%{}, %{}}, fn node, {id_mapping, new_nodes} ->
        case fun.(node) do
          :keep ->
            {id_mapping, new_nodes}

          %Axon{} = new_node ->
            {Map.put(id_mapping, node.id, new_node.output), Map.merge(new_nodes, new_node.nodes)}
        end
      end)

    nodes =
      axon.nodes
      |> Map.drop(Map.keys(id_mapping))
      |> Map.new(fn {id, node} ->
        {parent, :ok} =
          deep_map_reduce(node.parent, :ok, fn id, :ok ->
            {id_mapping[id] || id, :ok}
          end)

        {id, %{node | parent: parent}}
      end)
      |> Map.merge(new_nodes)

    %Axon{output: id_mapping[axon.output] || axon.output, nodes: nodes}
  end

  defp deep_map_reduce(%Axon{} = node, acc, fun) do
    fun.(node, acc)
  end

  defp deep_map_reduce(id, acc, fun) when is_integer(id) do
    fun.(id, acc)
  end

  defp deep_map_reduce(nodes, acc, fun) when is_list(nodes) do
    Enum.map_reduce(nodes, acc, &deep_map_reduce(&1, &2, fun))
  end

  defp deep_map_reduce(container, acc, fun) do
    Nx.Container.traverse(container, acc, &deep_map_reduce(&1, &2, fun))
  end
end
