defmodule Bumblebee.Layers do
  @moduledoc false

  import Nx.Defn

  @unsupported_activations [:gelu_approx_tanh, :gelu_approx_sigmoid]

  @pi :math.pi()

  @doc """
  Adds an activation layer.

  Handles all activations built into Axon, as well as several custom
  activations.

  ## Options

    * `:name` - layer name

  """
  def activation(%Axon{} = input, activation, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])
    name = opts[:name]

    if activation in @unsupported_activations do
      Axon.activation(input, &apply(__MODULE__, activation, [&1, &2]), name: name)
    else
      Axon.activation(input, activation, name: name)
    end
  end

  @doc """
  Implements the GeLU activation approximated with tanh.

  ## References

    * [Gaussian Error Linear Units (GeLUs)](https://arxiv.org/pdf/1606.08415.pdf)

  """
  defn gelu_approx_tanh(input, _opts \\ []) do
    0.5 * input *
      (1.0 + Nx.tanh(Nx.sqrt(2.0 / @pi) * (input + 0.044715 * Nx.pow(input, 3.0))))
  end

  @doc """
  Implements the GeLU activation approximated with sigmoid.

  Note that this approximation is less accurate than `gelu_approx_tanh/2`.

  ## References

    * [Gaussian Error Linear Units (GeLUs)](https://arxiv.org/pdf/1606.08415.pdf)

  """
  defn gelu_approx_sigmoid(input, _opts \\ []) do
    input * Nx.sigmoid(1.702 * input)
  end

  @doc """
  Computes relative attention bias.
  """
  def relative_attention_bias(query, key, attention_cache, offset, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        bidirectional: true,
        num_heads: 8,
        num_buckets: 32,
        max_distance: 128
      ])

    name = opts[:name]

    relative_position_buckets =
      Axon.layer(
        &compute_relative_position_buckets/4,
        [query, key, Axon.optional(attention_cache)],
        bidirectional: opts[:bidirectional],
        num_buckets: opts[:num_buckets],
        max_distance: opts[:max_distance]
      )

    bias =
      relative_position_buckets
      |> Axon.embedding(opts[:num_buckets], opts[:num_heads], name: name)
      |> Axon.transpose([2, 0, 1])
      |> Axon.nx(&Nx.new_axis(&1, 0))

    Axon.layer(
      fn bias, query, offset, _opts ->
        case offset do
          %Axon.None{} ->
            bias

          offset ->
            mask_shift = offset
            query_length = Nx.axis_size(query, 1)
            Nx.slice_along_axis(bias, mask_shift, query_length, axis: 2)
        end
      end,
      [bias, query, Axon.optional(offset)]
    )
  end

  defnp compute_relative_position_buckets(query, key, attention_cache, opts \\ []) do
    opts =
      keyword!(opts, mode: :inference, bidirectional: true, num_buckets: 32, max_distance: 128)

    {key_length, query_length} = key_query_lengths(query, key, attention_cache)

    context_position = Nx.iota({query_length, 1})
    memory_position = Nx.iota({1, key_length})
    relative_position = memory_position - context_position

    {num_buckets, relative_buckets, relative_position} =
      bidirectional_buckets(relative_position, opts[:num_buckets], opts[:bidirectional])

    max_exact = Nx.quotient(num_buckets, 2)
    is_small = Nx.less(relative_position, max_exact)

    relative_position_if_large =
      max_exact +
        Nx.log(relative_position / max_exact) / Nx.log(opts[:max_distance] / max_exact) *
          (num_buckets - max_exact)

    relative_position_if_large =
      Nx.min(
        relative_position_if_large,
        Nx.broadcast(num_buckets - 1, Nx.shape(relative_position_if_large))
      )
      |> Nx.as_type(:s64)

    relative_buckets + Nx.select(is_small, relative_position, relative_position_if_large)
  end

  deftransformp key_query_lengths(query, key, attention_cache) do
    case attention_cache do
      %Axon.None{} ->
        {Nx.axis_size(key, 1), Nx.axis_size(query, 1)}

      attention_cache ->
        key_length = Nx.axis_size(attention_cache.key, 1)
        {key_length, key_length}
    end
  end

  deftransformp bidirectional_buckets(relative_position, num_buckets, bidirectional) do
    relative_buckets = 0

    if bidirectional do
      num_buckets = div(num_buckets, 2)

      relative_buckets =
        Nx.add(relative_buckets, Nx.multiply(Nx.greater(relative_position, 0), num_buckets))

      relative_position = Nx.abs(relative_position)
      {num_buckets, relative_buckets, relative_position}
    else
      relative_position =
        relative_position
        |> Nx.min(Nx.broadcast(0, Nx.shape(relative_position)))
        |> Nx.negate()

      {num_buckets, relative_buckets, relative_position}
    end
  end

  @doc ~S"""
  Computes scaled dot-product attention for multiple attention heads.

  This is the core calculation behind multi-head attention, the projection
  layers should be applied on top of this layer.

  Given input sequences $Q, K, V \in R^{N \times d}$, where $N$ is the
  sequence length and $d$ is the head dimension, the scaled dot-product
  attention is defined as:

  $$
  Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  This operations is further batched across multiple heads and multiple
  input sequences.

  Intuitively scaled dot-product attention can be thought of as information
  retrieval, where for each sequence element in $Q$ the objective is
  to extract relevant context from sequence elements in $V$. In this
  analogy, $K$ is the summarization of information, while $V$ is the
  actual information. Then, assuming $Q$ and $K$ are embedded into a
  common space (which is the job of prior projection layers), the
  $QK^T$ dot product is a cosine similarity and gives us relevance
  weights for sequence elements in $V$.

  In case of self-attention, where $Q, K, V$ originate from the same
  sequence, the $QK^T$ weights indicate how much "each word attends
  to other words".

  ## Parameter Shapes

    * `query` - `{batch_size, sequence_length, num_heads, head_size}`
    * `key` - `{batch_size, kv_sequence_length, num_heads, head_size}`
    * `value` - `{batch_size, kv_sequence_length, num_heads, head_size}`
    * `key_mask` (optional) - `{batch_size, kv_sequence_length} | {batch_size, num_heads, sequence_length, kv_sequence_length}`
    * `head_mask` (optional) - `{num_heads}`
    * `bias` (optional) - `{batch_size | 1, num_heads | 1, sequence_length, kv_sequence_length}`
    * `offset` (optional) - `{}`

  ## Output Shape

    `{batch_size, sequence_length, num_heads, head_size}`

  ## Options

    * `:causal` - whether to apply causal mask to attention weights.
      This is typically used for next token prediction and it
      effectively makes each input token use information exclusively
      from prior tokens. Defaults to `false`

    * `:window_size` - when set, enables sliding window attention.
      Should be a `{left, right}` tuple with window size on each side

    * `:scale` - the scaling factor applied to the attention weights.
      Defaults to $\frac{1}{\sqrt{d}}$

    * `:dropout_rate` - the dropout rate for attention weights dropout.
      Defaults to `0.0`

  ## References

    * [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Figure 2 (left)

  """
  def attention(query, key, value, key_mask, head_mask, bias, offset, opts \\ []) do
    opts = Keyword.validate!(opts, [:window_size, :scale, causal: false, dropout_rate: 0.0])

    weights =
      Axon.layer(
        &attention_weights_impl/7,
        [
          query,
          key,
          Axon.optional(key_mask),
          Axon.optional(head_mask),
          Axon.optional(bias),
          Axon.optional(offset)
        ],
        causal: opts[:causal],
        window_size: opts[:window_size],
        scale: opts[:scale]
      )
      |> Axon.dropout(rate: opts[:dropout_rate])

    output = Axon.layer(&attention_output_impl/3, [weights, value], opts)

    {output, weights}
  end

  defnp attention_weights_impl(query, key, key_mask, head_mask, bias, offset, opts \\ []) do
    opts = keyword!(opts, [:window_size, mode: :inference, scale: true, causal: false])

    query = Nx.transpose(query, axes: [0, 2, 1, 3])
    key = Nx.transpose(key, axes: [0, 2, 1, 3])

    weights = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])

    scale =
      case opts[:scale] do
        nil ->
          depth = Nx.axis_size(query, -1)
          1 / Nx.as_type(Nx.sqrt(depth), Nx.type(query))

        scale ->
          scale
      end

    weights = weights * scale

    key_mask =
      case key_mask do
        %Axon.None{} ->
          Nx.broadcast(1, {1, 1, 1, 1})

        key_mask ->
          case Nx.rank(key_mask) do
            2 -> key_mask |> Nx.new_axis(1) |> Nx.new_axis(1)
            4 -> key_mask
          end
      end

    query_sequence_length = Nx.axis_size(query, 2)
    key_sequence_length = Nx.axis_size(key, 2)
    offset = ensure_offset(offset)

    causal_and_window_mask =
      case {opts[:causal], opts[:window_size]} do
        {false, nil} ->
          Nx.broadcast(1, {1, 1})

        {false, {left_size, right_size}} ->
          window_mask(query_sequence_length, key_sequence_length, offset, left_size, right_size)

        {true, nil} ->
          causal_mask(query_sequence_length, key_sequence_length, offset)

        {true, {left_size, _right_size}} ->
          window_mask(query_sequence_length, key_sequence_length, offset, left_size, 0)
      end
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)

    mask = key_mask and causal_and_window_mask

    bias =
      case bias do
        %Axon.None{} ->
          Nx.select(
            mask,
            Nx.tensor(0.0, type: Nx.type(query)),
            Nx.Constants.min_finite(Nx.type(query))
          )

        bias ->
          Nx.select(
            Nx.broadcast(mask, max_shape(mask, bias)),
            bias,
            Nx.Constants.min_finite(Nx.type(query))
          )
      end

    weights = weights + bias

    weights = Axon.Activations.softmax(weights, axis: -1)

    case head_mask do
      %Axon.None{} ->
        weights

      head_mask ->
        head_mask = Nx.reshape(head_mask, {1, :auto, 1, 1})
        Nx.multiply(weights, head_mask)
    end
  end

  defnp causal_mask(query_sequence_length, key_sequence_length, offset) do
    Nx.greater_equal(
      Nx.iota({query_sequence_length, 1}) + offset,
      Nx.iota({1, key_sequence_length})
    )
  end

  defnp window_mask(query_sequence_length, key_sequence_length, offset, left_size, right_size) do
    position_diff =
      Nx.subtract(
        Nx.iota({query_sequence_length, 1}) + offset,
        Nx.iota({1, key_sequence_length})
      )

    left_size >= position_diff and position_diff >= -right_size
  end

  defnp attention_output_impl(weights, value, _opts \\ []) do
    value = Nx.transpose(value, axes: [0, 2, 1, 3])
    out = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
    Nx.transpose(out, axes: [0, 2, 1, 3])
  end

  defnp ensure_offset(offset) do
    case offset do
      %Axon.None{} -> 0
      offset -> offset
    end
  end

  deftransformp max_shape(left, right) do
    Enum.zip_with(
      Tuple.to_list(Nx.shape(left)),
      Tuple.to_list(Nx.shape(right)),
      &max/2
    )
    |> List.to_tuple()
  end

  @doc """
  Adds a dense layer to the network.

  The kernel parameter is transposed with respect to `Axon.dense/3`.

  ## Options

    * `:name` - layer name

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `false`

  """
  def dense_transposed(%Axon{} = x, units, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: false
      ])

    kernel_shape = fn input_shape ->
      kernel_shape = Axon.Shape.dense_kernel(input_shape, units)

      # We expect a transposed kernel
      kernel_shape
      |> Tuple.to_list()
      |> Enum.reverse()
      |> List.to_tuple()
    end

    bias_shape = &Axon.Shape.dense_bias(&1, units)

    kernel = Axon.param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = Axon.param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], &dense_transposed_impl/4}
      else
        {[x, kernel], &dense_transposed_impl/3}
      end

    Axon.layer(op, inputs, name: opts[:name], op_name: :dense_transposed)
  end

  deftransformp dense_transposed_impl(x, kernel, bias \\ 0, _opts) do
    Nx.dot(x, [-1], kernel, [1])
    |> Nx.add(bias)
  end

  @doc """
  Adds a 1-dimensional convolution layer to the network.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`

  """
  def conv1d(%Axon{} = x, units, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true
      ])

    kernel_shape = fn input_shape ->
      {elem(input_shape, tuple_size(input_shape) - 1), units}
    end

    bias_shape = fn _ -> {units} end

    kernel = Axon.param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = Axon.param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], &conv1d_impl/4}
      else
        {[x, kernel], &conv1d_impl(&1, &2, 0, &3)}
      end

    Axon.layer(op, inputs, name: opts[:name], op_name: :conv1d)
  end

  defnp conv1d_impl(input, kernel, bias, _opts \\ []) do
    input
    |> Nx.dot([Nx.rank(input) - 1], [], kernel, [0], [])
    |> Nx.add(bias)
  end

  @doc """
  Adds a scaling layer to the network.

  The scaling layer scales inputs by a learned scale parameter.

  ## Options

    * `:name` - layer name

    * `:scale_initializer` - initializer for the scale parameter

    * `:channel_index` - index of the axis to scale. Defaults to the
      last axis

  """
  def scale(%Axon{} = x, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        scale_initializer: Axon.Initializers.full(1.0e-6),
        channel_index: -1
      ])

    name = opts[:name]
    scale_initializer = opts[:scale_initializer]
    channel_index = opts[:channel_index]

    scale_shape = fn input_shape ->
      rank = tuple_size(input_shape)
      channel_index = rem(rank + channel_index, rank)
      out_channels = elem(input_shape, channel_index)
      {out_channels}
    end

    scale_param = Axon.param("scale", scale_shape, initializer: scale_initializer)

    Axon.layer(
      fn input, scale, _opts ->
        channel_index = Nx.axis_index(input, channel_index)
        shape = Tuple.duplicate(1, Nx.rank(input)) |> put_elem(channel_index, :auto)
        scale = Nx.reshape(scale, shape)
        Nx.multiply(input, scale)
      end,
      [x, scale_param],
      name: name,
      op_name: :scale
    )
  end

  @doc """
  Adds a drop-path layer to the network for stochastic depth.

  ## Options

    * `:name` - layer name

    * `:rate` - drop path rate

  ## References

    * [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)

  """
  def drop_path(%Axon{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, :seed, rate: 0.0])
    seed = Keyword.get_lazy(opts, :seed, fn -> :erlang.system_time() end)

    key_state =
      Axon.param("key", fn _ -> {2} end,
        type: {:u, 32},
        initializer: fn _, _ -> Nx.Random.key(seed) end
      )

    if opts[:rate] > 0.0 do
      Axon.layer(&drop_path_impl/3, [input, key_state],
        name: opts[:name],
        op_name: :drop_path,
        rate: opts[:rate]
      )
    else
      input
    end
  end

  deftransformp drop_path_impl(x, prng_key, opts \\ []) do
    opts = Keyword.validate!(opts, rate: 0.0, mode: :train)
    rate = opts[:rate]

    case opts[:mode] do
      :train ->
        keep_prob = 1 - rate
        shape = Tuple.duplicate(1, Nx.rank(x)) |> put_elem(0, Nx.axis_size(x, 0))

        {rand, next_key} = Nx.Random.uniform(prng_key, shape: shape)

        bernoulli_noise =
          keep_prob
          |> Nx.add(rand)
          |> Nx.floor()

        out =
          x
          |> Nx.divide(keep_prob)
          |> Nx.multiply(bernoulli_noise)

        %Axon.StatefulOutput{output: out, state: %{"key" => next_key}}

      _mode ->
        x
    end
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
  def take_token(%Axon{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, :axis, index: 0])

    Axon.nx(
      input,
      fn x ->
        x
        |> Nx.slice_along_axis(opts[:index], 1, axis: opts[:axis])
        |> Nx.squeeze(axes: [opts[:axis]])
      end,
      name: opts[:name]
    )
  end

  @doc """
  Implements position masking for embedded patches of visual inputs.

  This layer expects computed patch embeddings and an optional mask.
  If the mask is not specified, it will skip masking altogether.

  ## Options

    * `:name` - layer name

  """
  def apply_vision_patch_mask(%Axon{} = embeddings, patch_mask, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])
    name = opts[:name]

    mask_token_shape = fn embeddings_shape, _ ->
      hidden_size = elem(embeddings_shape, 2)
      {1, 1, hidden_size}
    end

    mask_token = Axon.param("mask_token", mask_token_shape, initializer: :zeros)

    if_present patch_mask do
      Axon.layer(
        fn embeddings, patch_mask, mask_tokens, _opts ->
          hidden_size = Nx.axis_size(embeddings, 2)
          batch_size = Nx.axis_size(embeddings, 0)
          sequence_length = Nx.axis_size(embeddings, 1)
          mask_tokens = Nx.broadcast(mask_tokens, {batch_size, sequence_length, hidden_size})

          mask =
            patch_mask
            |> Nx.new_axis(-1)
            |> Nx.broadcast({batch_size, sequence_length, hidden_size})

          Nx.select(mask, mask_tokens, embeddings)
        end,
        [embeddings, patch_mask, mask_token],
        name: name,
        op_name: :apply_patch_mask
      )
    else
      embeddings
    end
  end

  @doc """
  Splits the hidden dimension into the given number of attention heads.

  In other words, the input with shape `{batch_size, sequence_length, hidden_size}`
  is reshaped to `{batch_size, sequence_length, num_heads, *}`.
  """
  def split_heads(states, num_heads) do
    Axon.nx(states, fn states ->
      batch_size = Nx.axis_size(states, 0)
      sequence_length = Nx.axis_size(states, 1)
      new_shape = {batch_size, sequence_length, num_heads, :auto}
      Nx.reshape(states, new_shape)
    end)
  end

  @doc """
  Splits the input node with shape `{batch_size, sequence_length, 2}` into
  two nodes with shape `{batch_size, sequence_length}`.
  """
  def split_pair(%Axon{} = x) do
    left = Axon.nx(x, & &1[[.., .., 0]])
    right = Axon.nx(x, & &1[[.., .., 1]])
    {left, right}
  end

  @doc """
  Adds a layer to the network which flattens the leading axes of the
  input.
  """
  def flatten_leading(%Axon{} = x) do
    Axon.nx(x, fn x ->
      shape =
        x
        |> Nx.shape()
        |> Tuple.delete_at(0)
        |> put_elem(0, :auto)

      Nx.reshape(x, shape)
    end)
  end

  @doc """
  Adds a layer to the network which flattens the trailing axes of the
  input.
  """
  def flatten_trailing(%Axon{} = x) do
    Axon.nx(x, fn x ->
      shape = Nx.shape(x)
      rank = tuple_size(shape)

      shape =
        shape
        |> Tuple.delete_at(rank - 1)
        |> put_elem(rank - 2, :auto)

      Nx.reshape(x, shape)
    end)
  end

  @doc """
  Adds a pixel rearrangement layer to the network.

  Rearranges elements in a tensor of shape `{*, H, W, C × r^2}` to a
  tensor of shape `{*, H × r, W × r, C}`, where r is an upscale factor.

  This is useful for implementing efficient sub-pixel convolution
  with a stride of `1 / r`.

  ## Options

    * `:name` - layer name

  """
  def pixel_shuffle(input, upscale_factor, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])

    Axon.layer(&pixel_shuffle_impl/2, [input],
      name: opts[:name],
      op_name: :pixel_shuffle,
      upscale_factor: upscale_factor
    )
  end

  deftransformp pixel_shuffle_impl(input, opts \\ []) do
    opts = Keyword.validate!(opts, [:upscale_factor, mode: :inference])
    upscale_factor = opts[:upscale_factor]

    {batch, [height, width, channels]} =
      input
      |> Nx.shape()
      |> Tuple.to_list()
      |> Enum.split(-3)

    out_height = height * upscale_factor
    out_width = width * upscale_factor
    out_channels = div(channels, upscale_factor * upscale_factor)

    x =
      Nx.reshape(
        input,
        List.to_tuple(batch ++ [height, width, out_channels, upscale_factor, upscale_factor])
      )

    {batch_axes, [height_axis, width_axis, out_channels_axis, upscale_axis1, upscale_axis2]} =
      x
      |> Nx.axes()
      |> Enum.split(-5)

    x
    |> Nx.transpose(
      axes:
        batch_axes ++ [height_axis, upscale_axis1, width_axis, upscale_axis2, out_channels_axis]
    )
    |> Nx.reshape(List.to_tuple(batch ++ [out_height, out_width, out_channels]))
  end

  @doc """
  Adds a layer that computes cosine similarity between the inputs.
  """
  def cosine_similarity(x, y) do
    Axon.layer(&cosine_similarity_impl/3, [x, y], op_names: :cosine_similarity)
  end

  defnp cosine_similarity_impl(x, y, _opts \\ []) do
    Bumblebee.Utils.Nx.cosine_similarity(x, y)
  end

  @doc """
  Unwraps a tuple result from `Axon` node into separate nodes.
  """
  def unwrap_tuple(%Axon{} = input, size) do
    for i <- 0..(size - 1) do
      Axon.nx(input, &elem(&1, i))
    end
    |> List.to_tuple()
  end

  @doc """
  Adds a default layer to handle optional nodes.

  This layer evaluates to the result of `x` if present, otherwise
  falls back to the result of the default node.

  ## Examples

      input_ids = Axon.input("input_ids")
      attention_mask = Axon.input("attention_mask", optional: true)

      attention_mask =
        Bumblebee.Layers.default attention_mask do
          Axon.nx(input_ids, &Nx.broadcast(1, &1))
        end

  """
  def default(%Axon{} = x, do: default) do
    Axon.layer(
      fn x, default, _ ->
        case x do
          %Axon.None{} -> default
          _ -> x
        end
      end,
      [Axon.optional(x), Axon.optional(default)],
      op_name: :default
    )
  end

  @doc """
  Adds a conditional layer.

  This layer evaluates to either branch, depending on whether the
  optional `condition` value is present or missing.

  The branches can be either `Axon` nodes or `Nx.Container`s with
  `Axon` nodes and the same structure. If containers are given, this
  function also returns a matching container.

  ## Examples

      {hidden_state, cross_attention} =
        Bumblebee.Layers.if_present encoder_hidden_state do
          ...
          {hidden_state, cross_attention}
        else
          {hidden_state, Bumblebee.Layers.none()}
        end

  """
  def if_present(%Axon{} = condition, blocks) do
    on_true = Keyword.fetch!(blocks, :do)
    on_false = blocks[:else]

    case {on_true, on_false} do
      {%Axon{}, %Axon{}} ->
        if_present_layer(condition, on_true, on_false)

      {%Axon{}, nil} ->
        if_present_layer(condition, on_true, none())

      _ ->
        on_false = on_false || Bumblebee.Utils.Axon.container_map(on_true, fn _ -> none() end)

        Bumblebee.Utils.Axon.container_zip_with(on_true, on_false, fn left, right ->
          if_present_layer(condition, left, right)
        end)
    end
  end

  defp if_present_layer(condition, on_true, on_false) do
    Axon.layer(
      fn condition, on_true, on_false, _ ->
        case condition do
          %Axon.None{} -> on_false
          _ -> on_true
        end
      end,
      [Axon.optional(condition), Axon.optional(on_true), Axon.optional(on_false)],
      op_name: :if_present
    )
  end

  @doc """
  Returns an Axon layer that resolves to `%Axon.None{}`.
  """
  def none() do
    Axon.layer(fn _opts -> %Axon.None{} end, [], op_name: :none)
  end

  @doc """
  Adds a layer that passes the input through only if the given global
  layer option is set.
  """
  def global_opt_in(%Axon{} = input, global_option_name) do
    Axon.layer(
      fn input, opts ->
        if opts[global_option_name] do
          input
        else
          %Axon.None{}
        end
      end,
      [input],
      op_name: :global_opt_in,
      global_options: [global_option_name]
    )
  end

  @doc """
  Appends tuple element to the node result.
  """
  def append(%Axon{} = tuple, %Axon{} = x) do
    Axon.layer(
      fn tuple, x, _ ->
        Tuple.insert_at(tuple, tuple_size(tuple), x)
      end,
      [tuple, x],
      op_name: :append
    )
  end

  @doc """
  Replaces tuple element in the node result.
  """
  def replace(%Axon{} = tuple, idx, %Axon{} = x) do
    Axon.layer(fn tuple, x, _ -> tuple_replace(tuple, idx, x) end, [tuple, x], op_name: :replace)
  end

  defp tuple_replace(tuple, index, value) when index < 0 do
    tuple_replace(tuple, tuple_size(tuple) + index, value)
  end

  defp tuple_replace(tuple, index, value) do
    put_elem(tuple, index, value)
  end

  @doc """
  Builds an `Axon` container with the given outputs.

  All values are wrapped with `Axon.optional/2`, so if any of them is
  missing, it gets returned as `%Axon.None{}`.

  Also, guards known optional outputs behind a global layer option
  using `global_opt_in/2`.
  """
  @spec output(map()) :: Axon.t()
  def output(outputs) do
    outputs
    |> Map.new(fn {key, %Axon{} = val} ->
      {key, val |> maybe_opt_in_output(key) |> Axon.optional()}
    end)
    |> Axon.container()
  end

  @opt_in_outputs %{
    :hidden_states => :output_hidden_states,
    :attentions => :output_attentions
  }

  defp maybe_opt_in_output(%Axon{} = input, key) do
    if option_name = @opt_in_outputs[key] do
      global_opt_in(input, option_name)
    else
      input
    end
  end

  @doc """
  Computes a 1-full mask matching the first two dimensions of `input`
  (batch size and sequence length).
  """
  def default_attention_mask(%Axon{} = input) do
    Axon.nx(input, fn input ->
      batch_size = Nx.axis_size(input, 0)
      sequence_length = Nx.axis_size(input, 1)
      Nx.broadcast(1, {batch_size, sequence_length})
    end)
  end

  @doc """
  Computes increasing position ids matching the first two dimensions
  of `input` (batch size and sequence length).

  ## Options

    * `:offset` - the index of the first position. Defaults to `0`

  """
  def default_position_ids(%Axon{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, offset: 0)
    offset = opts[:offset]

    Axon.nx(input, fn input ->
      batch_size = Nx.axis_size(input, 0)
      sequence_length = Nx.axis_size(input, 1)
      Nx.iota({batch_size, sequence_length}, axis: -1) |> Nx.add(offset)
    end)
  end

  @doc """
  Computes 0-full mask matching the first two dimensions of `input`
  (batch size and sequence length).
  """
  def default_token_type_ids(%Axon{} = input) do
    Axon.nx(input, fn input ->
      batch_size = Nx.axis_size(input, 0)
      sequence_length = Nx.axis_size(input, 1)
      Nx.broadcast(0, {batch_size, sequence_length})
    end)
  end

  @doc """
  Computes 0-full bounding box for document-understanding models.
  """
  def default_bounding_box(%Axon{} = input) do
    Axon.nx(input, fn input ->
      batch_size = Nx.axis_size(input, 0)
      sequence_length = Nx.axis_size(input, 1)
      Nx.broadcast(0, {batch_size, sequence_length, 4})
    end)
  end

  @doc """
  Shifts the given input ids by removing the last token and prepending
  the given start token.

  Some models use this technique to generate default decoder input ids.
  """
  def shift_tokens_right(%Axon{} = input_ids, decoder_start_token_id) do
    Axon.nx(input_ids, fn input_ids ->
      batch_size = Nx.axis_size(input_ids, 0)
      sequence_length = Nx.axis_size(input_ids, 1)
      start_ids = Nx.broadcast(decoder_start_token_id, {batch_size, 1})

      if sequence_length == 1 do
        start_ids
      else
        Nx.concatenate([start_ids, input_ids[[.., 0..-2//1]]], axis: 1)
      end
    end)
  end

  @doc """
  Returns a node with parameterized embeddings.

  ## Options

    * `:name` - layer name

    * `:initializer` - initializer for the embeddings. Defaults to
      `:zeros`

  """
  def learned_embeddings(num_embeddings, embedding_size, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, initializer: :zeros])

    name = opts[:name]

    embeddings =
      Axon.param("embeddings", fn -> {num_embeddings, embedding_size} end,
        initializer: opts[:initializer]
      )

    Axon.layer(
      fn embeddings, _opts -> Nx.new_axis(embeddings, 0) end,
      [embeddings],
      name: name,
      op_name: :learned_embeddings
    )
  end

  @doc """
  Concatenates sequence embeddings, automatically broadcasting batch
  dimension when necessary.

  ## Options

    * `:name` - layer name

  """
  def concatenate_embeddings(inputs, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])

    name = opts[:name]

    Axon.layer(
      fn inputs, _opts ->
        inputs = Tuple.to_list(inputs)

        batch_size =
          inputs
          |> Enum.map(&Nx.axis_size(&1, 0))
          |> Enum.max()

        inputs =
          Enum.map(inputs, fn input ->
            new_shape = input |> Nx.shape() |> put_elem(0, batch_size)
            Nx.broadcast(input, new_shape)
          end)

        Nx.concatenate(inputs, axis: 1)
      end,
      [Axon.container(List.to_tuple(inputs))],
      name: name,
      op_name: :concatenate_embeddings
    )
  end

  @doc """
  Adds an RMS Normalization layer to the network.

  ## Options

    * `:name` - layer name

    * `:initializer` - initializer for the standard deviation parameter.
      Defaults to `:ones`

    * `:channel_index` - input feature index used for calculating
      variance. Defaults to `-1`

    * `:epsilon` - numerical stability term

    * `:shift` - numeric shift in the scaling expression. Defaults to
      `0.0`

    * `:upcast` - adds explicit type casting to make sure the norm
      is computed in high numerical precision. Either of:

      * `:normalization` (default) - upcasts only the input normalization
        part

      * `:all` - upcasts both input normalization and the scaling
        expression

  """
  # TODO: Add to Axon
  def rms_norm(input, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        shift: 0.0,
        channel_index: -1,
        epsilon: 1.0e-6,
        upcast: :normalization,
        initializer: :ones
      ])

    impl =
      case opts[:upcast] do
        :normalization ->
          &rms_norm_impl_upcast_normalization/3

        :all ->
          &rms_norm_impl_upcast_all/3

        other ->
          raise ArgumentError,
                "expected :upcast to be either :all or :normalization, got: #{other}"
      end

    weight =
      Axon.param("weight", &Axon.Shape.norm_param(&1, opts[:channel_index]),
        initializer: opts[:initializer]
      )

    Axon.layer(impl, [input, weight],
      name: opts[:name],
      shift: opts[:shift],
      epsilon: opts[:epsilon],
      op_name: :rms_norm
    )
  end

  defnp rms_norm_impl_upcast_normalization(input, weight, opts \\ []) do
    opts = keyword!(opts, shift: 0.0, epsilon: 1.0e-6, channel_index: -1, mode: :train)

    normalized_input =
      input
      |> Nx.as_type(:f32)
      |> rms_normalize(opts)
      |> Nx.as_type(Nx.type(input))

    normalized_input * (opts[:shift] + weight)
  end

  defnp rms_norm_impl_upcast_all(input, weight, opts \\ []) do
    opts = keyword!(opts, shift: 0.0, epsilon: 1.0e-6, channel_index: -1, mode: :train)

    input = Nx.as_type(input, :f32)
    weight = Nx.as_type(weight, :f32)

    normalized_input = rms_normalize(input, opts)

    normalized_input * (opts[:shift] + weight)
  end

  defnp rms_normalize(input, opts) do
    variance =
      input
      |> Nx.pow(2)
      |> Nx.mean(axes: [opts[:channel_index]], keep_axes: true)

    input * Nx.rsqrt(variance + opts[:epsilon])
  end

  @doc """
  Adds a rotary embedding layer to the network.
  """
  def rotary_embedding(query, key, position_ids, attention_mask, size, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, :scaling_strategy, max_positions: 2048, base: 10_000])

    output =
      Axon.layer(
        &apply_rotary_embedding/5,
        [query, key, position_ids, Axon.optional(attention_mask)],
        [size: size] ++ opts
      )

    unwrap_tuple(output, 2)
  end

  deftransformp create_sinusoidal_positions(
                  sequence_length,
                  max_positions,
                  size,
                  base,
                  scaling_strategy
                ) do
    position = Nx.iota({sequence_length})

    range = Nx.iota({div(size, 2)}) |> Nx.multiply(2) |> Nx.divide(size)

    case scaling_strategy do
      %{type: :linear, factor: factor} ->
        inv_frequency = inv_frequency(base, range)
        position = Nx.divide(position, factor)
        positions_cos_sin(position, inv_frequency)

      %{type: :dynamic, factor: factor} when sequence_length > max_positions ->
        base =
          base
          |> Nx.multiply(factor * sequence_length / max_positions - (factor - 1))
          |> Nx.pow(size / (size - 2))

        inv_frequency = inv_frequency(base, range)
        positions_cos_sin(position, inv_frequency)

      %{
        type: :longrope,
        short_factor: short_factor,
        long_factor: long_factor,
        original_max_positions: original_max_positions
      } ->
        factor =
          if sequence_length > original_max_positions do
            Nx.tensor(long_factor, type: :f32)
          else
            Nx.tensor(short_factor, type: :f32)
          end

        scale = max_positions / original_max_positions

        cos_sin_factor =
          if scale <= 1.0 do
            1.0
          else
            Nx.divide(Nx.log(scale), Nx.log(original_max_positions))
            |> Nx.add(1)
            |> Nx.sqrt()
          end

        inv_frequency = inv_frequency(base, range) |> Nx.divide(factor)
        {cos, sin} = positions_cos_sin(position, inv_frequency)
        {Nx.multiply(cos, cos_sin_factor), Nx.multiply(sin, cos_sin_factor)}

      %{
        type: :llama3,
        factor: factor,
        low_frequency_factor: low_frequency_factor,
        high_frequency_factor: high_frequency_factor,
        original_max_positions: original_max_positions
      } ->
        inv_frequency = inv_frequency(base, range)

        inv_frequency =
          llama3_inv_frequency(
            inv_frequency,
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_positions
          )

        positions_cos_sin(position, inv_frequency)

      _other ->
        inv_frequency = inv_frequency(base, range)
        positions_cos_sin(position, inv_frequency)
    end
  end

  defnp llama3_inv_frequency(
          inv_frequency,
          factor,
          low_frequency_factor,
          high_frequency_factor,
          original_max_positions
        ) do
    low_frequency_wavelength = original_max_positions / low_frequency_factor
    high_frequency_wavelength = original_max_positions / high_frequency_factor

    # Vectorize to enable cleaner conditional
    inv_frequency = Nx.vectorize(inv_frequency, :range)

    wavelength = 2 * Nx.Constants.pi() / inv_frequency

    inv_frequency =
      cond do
        wavelength < high_frequency_wavelength ->
          inv_frequency

        wavelength > low_frequency_wavelength ->
          inv_frequency / factor

        true ->
          # Interpolation between the two cases above

          smooth_factor =
            (original_max_positions / wavelength - low_frequency_factor) /
              (high_frequency_factor - low_frequency_factor)

          (1 - smooth_factor) * inv_frequency / factor + smooth_factor * inv_frequency
      end

    Nx.devectorize(inv_frequency)
  end

  defnp inv_frequency(base, range) do
    frequency = Nx.pow(base, range)
    1.0 / frequency
  end

  defnp positions_cos_sin(position, inv_frequency) do
    angle = Nx.outer(position, inv_frequency)
    angle = Nx.concatenate([angle, angle], axis: -1)
    {Nx.cos(angle), Nx.sin(angle)}
  end

  defnp apply_rotary_embedding(query, key, position_ids, attention_mask, opts \\ []) do
    opts =
      keyword!(opts, [
        :size,
        :scaling_strategy,
        mode: :inference,
        max_positions: 2048,
        base: 10_000
      ])

    # When decoding with cache position_ids may be a partial sequence,
    # but in that case we always have full-length attention mask
    sequence_length =
      case attention_mask do
        %Axon.None{} -> Nx.axis_size(position_ids, 1)
        _other -> Nx.axis_size(attention_mask, 1)
      end

    {cos, sin} =
      create_sinusoidal_positions(
        sequence_length,
        opts[:max_positions],
        opts[:size],
        opts[:base],
        opts[:scaling_strategy]
      )

    position_ids = Nx.as_type(position_ids, :s64)

    cos = cos |> Nx.take(position_ids) |> Nx.new_axis(2) |> Nx.as_type(Nx.type(query))
    sin = sin |> Nx.take(position_ids) |> Nx.new_axis(2) |> Nx.as_type(Nx.type(query))

    rotated_query = query * cos + rotate_half(query) * sin
    rotated_key = key * cos + rotate_half(key) * sin

    {rotated_query, rotated_key}
  end

  defnp rotate_half(x) do
    size = div(Nx.axis_size(x, -1), 2)
    x1 = x[[.., .., .., 0..(size - 1)//1]]
    x2 = x[[.., .., .., size..-1//1]]
    Nx.concatenate([-x2, x1], axis: -1)
  end

  @doc """
  Adds a repeat layer to the network.

  ## Options

    * `:name` - layer name

    * `:axis` - the axis to repeat along. Defaults to `-1`

  """
  def repeat_interleave(x, times, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, axis: -1])

    Axon.layer(
      fn x, opts ->
        axis = Nx.axis_index(x, opts[:axis])
        Bumblebee.Utils.Nx.repeat_interleave(x, times, axis: axis)
      end,
      [x],
      opts
    )
  end
end
