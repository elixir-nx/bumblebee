defmodule Bumblebee.Layers do
  @moduledoc """
  Custom layers.
  """

  import Nx.Defn

  @doc """
  Converts mask to bias.
  """
  defn attention_bias(attention_mask, _params) do
    attention_mask =
      attention_mask
      |> Nx.new_axis(-3)
      |> Nx.new_axis(-2)

    mask = Nx.greater(attention_mask, 0)
    on_true = Nx.broadcast(0, attention_mask)
    on_false = Nx.broadcast(-1.0e10, attention_mask)

    Nx.select(mask, on_true, on_false)
  end

  @doc """
  Computes attention weights.
  """
  defn dot_product_attention_weights(query, key, bias, _params, opts \\ []) do
    opts = keyword!(opts, [:axes])
    axes = transform(opts[:axes], fn
      nil ->
        Enum.to_list(1..Nx.rank(key) - 3)
      axes ->
        axes
    end)

    depth = elem(Nx.shape(query), Nx.rank(query) - 1)
    n = transform(key, &Nx.rank(&1))

    {_batch_dims, qk_perm} = transform({axes, n}, fn {axes, n} ->
      batch_dims = Enum.to_list(0..n - 1) -- (axes ++ [n - 1])
      qk_perm = batch_dims ++ axes ++ [n - 1]
      {batch_dims, qk_perm}
    end)

    key = Nx.transpose(key, axes: qk_perm)
    query = Nx.transpose(query, axes: qk_perm)
    scaled_query = Nx.divide(query, Nx.sqrt(depth))

    # TODO: Batch dims are wrong :(
    weights = Nx.dot(scaled_query, [n - 1], [0, 1], key, [n - 1], [0, 1])
    weights = weights + bias

    norm_dims = transform({axes, Nx.rank(weights)}, fn {axes, weights_ndim} ->
      Enum.to_list((weights_ndim - length(axes))..(weights_ndim - 1))
    end)

    Axon.Activations.softmax(weights, axis: norm_dims)
  end

  @doc """
  Computes attention outputs.
  """
  defn dot_product_attention_output(attn_weights, value, _params) do
    value = Nx.transpose(value, axes: [0, 2, 1, 3])
    out = Nx.dot(attn_weights, [3], [0, 1], value, [2], [0, 1])
    Nx.transpose(out, axes: [0, 2, 1, 3])
  end
end