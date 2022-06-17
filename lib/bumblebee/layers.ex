defmodule Bumblebee.Layers do
  @moduledoc """
  Custom layers.
  """

  import Nx.Defn

  @doc """
  Converts attention mask to bias.
  """
  defn attention_bias(attention_mask, _opts \\ []) do
    attention_mask =
      attention_mask
      |> Nx.new_axis(-2)
      |> Nx.new_axis(-2)

    Nx.select(attention_mask > 0, 0, -1.0e10)
  end

  @doc """
  Computes attention weights.
  """
  defn attention_weights(query, key, bias, _opts \\ []) do
    key = Nx.transpose(key, axes: [0, 2, 1, 3])
    query = Nx.transpose(query, axes: [0, 2, 1, 3])

    depth = Nx.axis_size(query, -1)
    scaled_query = query / Nx.sqrt(depth)

    weights = Nx.dot(scaled_query, [3], [0, 1], key, [3], [0, 1])
    weights = weights + bias
    Axon.Activations.softmax(weights, axis: -1)
  end

  @doc """
  Computes attention outputs.
  """
  defn attention_output(attention_weights, value, _opts \\ []) do
    value = Nx.transpose(value, axes: [0, 2, 1, 3])
    out = Nx.dot(attention_weights, [3], [0, 1], value, [2], [0, 1])
    Nx.transpose(out, axes: [0, 2, 1, 3])
  end

  @doc """
  Adds a dense layer to the network.

  The kernel parameter is transposed with respect to `Axon.dense/3`.

  ## Options

    * `:name` - layer name

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`

  """
  def dense_transposed_layer(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, kernel_initializer: :glorot_uniform])

    kernel_shape = Axon.Shape.dense_kernel(parent_shape, units)
    output_shape = Axon.Shape.dense(parent_shape, units)

    # We expect a transposed kernel
    kernel_shape =
      kernel_shape
      |> Tuple.to_list()
      |> Enum.reverse()
      |> List.to_tuple()

    kernel = Axon.param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    op = fn x, kernel, _opts ->
      Nx.dot(x, [-1], kernel, [1])
    end

    Axon.layer(op, [x, kernel],
      name: opts[:name],
      shape: output_shape,
      op_name: :dense_transposed
    )
  end
end
