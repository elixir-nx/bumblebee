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
  Applies head mask to the attention weights for the given layer.
  """
  defn apply_layer_head_mask(attention_weights, layer_head_mask, _opts \\ []) do
    layer_head_mask = Nx.reshape(layer_head_mask, {1, :auto, 1, 1})
    Nx.multiply(attention_weights, layer_head_mask)
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
  Updates cache for fast autoregressive decoding.
  """
  defn update_cache(query_states, key_states, value_states, attention_mask, cache, _opts \\ []) do
    %{key: cached_key, value: cached_value, index: index} = cache
    {batch, max_length, _num_heads, _head_dim} = Nx.shape(cached_value)
    indices = [0, 0, Nx.as_type(index, {:s, 64}), 0]
    key = Nx.put_slice(cached_key, indices, key_states)
    value = Nx.put_slice(cached_value, indices, value_states)
    num_updated_cache_vectors = Nx.axis_size(query_states, 1)
    index = index + num_updated_cache_vectors

    pad_mask =
      Nx.iota({max_length})
      |> Nx.less(Nx.add(index, num_updated_cache_vectors))
      |> Nx.broadcast({batch, 1, num_updated_cache_vectors, max_length})

    attention_mask = Nx.logical_and(pad_mask, attention_mask)
    updated_cache = %{key: key, value: value, index: index}
    {key, value, attention_mask, updated_cache}
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

  @doc """
  Adds a scaling layer to the network.

  The scaling layer scales inputs by a learned scale parameter.

  ## Options

    * `:name` - layer name

    * `:scale_name` - scale parameter name

    * `:scale_init_value` - initial value of scale parameter

  """
  def scale_layer(%Axon{output_shape: parent_shape} = x, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        scale_name: "layer_scale_parameter",
        scale_init_value: 1.0e-6,
        channel_index: 1
      ])

    name = opts[:name]
    scale_name = opts[:scale_name]
    scale_init_value = opts[:scale_init_value]
    channel_index = opts[:channel_index]

    out_channels = elem(parent_shape, channel_index)

    scale_param =
      Axon.param(scale_name, {out_channels},
        initializer: fn shape, _ ->
          Nx.broadcast(scale_init_value, shape)
        end
      )

    Axon.layer(fn input, scale, _opts -> Nx.multiply(input, scale) end, [x, scale_param],
      name: name,
      op_name: :scale
    )
  end

  @doc """
  Adds a drop-path layer to the network for stochastic
  depth.

  ## Options

    * `:name` - layer name

    * `:rate` - drop path rate
  """
  def drop_path_layer(%Axon{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, rate: 0.0])

    if opts[:rate] > 0.0 do
      Axon.layer(&drop_path/2, [input], name: opts[:name], rate: opts[:rate])
    else
      input
    end
  end

  defnp drop_path(x, opts \\ []) do
    opts = keyword!(opts, rate: 0.0, mode: :train)

    transform({x, opts[:rate], opts[:mode]}, fn
      {x, rate, :train} when Elixir.Kernel.!=(rate, 0.0) ->
        keep_prob = 1 - rate
        shape = elem(Nx.shape(x), 0)

        random_tensor =
          keep_prob
          |> Nx.add(Nx.random_uniform(shape))
          |> Nx.floor()

        x |> Nx.divide(keep_prob) |> Nx.multiply(random_tensor)

      {x, _rate, _mode} ->
        x
    end)
  end

  @doc """
  Takes the first element along the given axis.

  This is a common operation in many architectures. It reduces
  dimensionality by dropping the given axis.

  ## Options

    * `:name` - layer name
    
    * `:axis` - axis to slice token from

    * `:index` - index to slice head. Defaults to 0
  """
  def take_token_layer(%Axon{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:axis, :name, index: 0])

    input
    |> Axon.nx(fn x ->
      x
      |> Nx.slice_along_axis(opts[:index], 1, axis: opts[:axis])
      |> Nx.squeeze(axes: [opts[:axis]])
    end)
  end

  @doc """
  Implements position masking for embedded patches of visual
  inputs.

  This layer expects computed patch embeddings and an optional
  mask. If the mask is not specified, it will skip position masking
  altogether.

  ## Options

    * `:name` - layer name

    * `:mask_size` - size of mask
  """
  def vision_position_mask_layer(%Axon{output_shape: shape} = embeds, bool_masked_pos, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, :mask_size])
    name = opts[:name]

    mask_size = elem(shape, 2)
    mask_token = Axon.param("mask_token", {1, 1, mask_size}, initializer: :zeros)

    Axon.layer(
      fn embeds, bool_mask, toks, _opts ->
        if bool_mask do
          batch_size = Nx.axis_size(embeds, 0)
          seq_len = Nx.axis_size(embeds, 1)
          mask_tokens = Nx.broadcast(toks, {batch_size, seq_len, mask_size})
          mask = bool_mask |> Nx.new_axis(-1) |> Nx.broadcast({batch_size, seq_len, mask_size})
          Nx.select(mask, mask_tokens, embeds)
        else
          embeds
        end
      end,
      [embeds, bool_masked_pos, mask_token],
      name: name

  @doc """
  A conditional layer which checks the presence of
  a value and returns another graph value otherwise.

  Similar to Haskell's `maybe` function.
  """
  def maybe_layer(%Axon{} = maybe_value, %Axon{} = default_value, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])

    Axon.layer(
      fn
        nil, default_value, _opts -> default_value
        maybe_value, _, _opts -> maybe_value
      end,
      [maybe_value, default_value],
      name: opts[:name],
      op_name: :maybe
    )
  end
end
