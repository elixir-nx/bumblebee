defmodule Bumblebee.Layers.Mamba do
  @moduledoc false

  # Layers implementing the Mamba-2 state-space mixer, as used by hybrid
  # models such as Nemotron-H.
  #
  # The mixer replaces self-attention with a linear recurrence over the
  # sequence. For every head, the state is a matrix of shape
  # `{head_size, state_size}`, updated as
  #
  #     state_t = exp(dt_t * A) * state_{t - 1} + dt_t * B_t x_t'
  #     y_t = state_t C_t + D x_t
  #
  # where `A` and `D` are per-head scalars, while `dt`, `B` and `C` are
  # computed from the input (which is what makes the recurrence selective).
  #
  # ## Implementation notes
  #
  # The recurrence is computed with the chunked (SSD) algorithm. The
  # sequence is split into chunks and within each chunk the recurrence is
  # unrolled into a quadratic, attention-like form, which parallelizes
  # well, while the chunk states are combined with a much shorter
  # recurrence. Note that the decay factors are accumulated in log space,
  # so this is numerically equivalent to the sequential form.

  alias Bumblebee.Layers

  import Bumblebee.Utils.Model, only: [join: 2]

  @doc """
  Adds a Mamba-2 mixer to the network.

  Returns the mixer output and the updated cache.

  ## Options

    * `:hidden_size` (required) - the dimensionality of the input and
      output

    * `:num_heads` (required) - the number of state-space heads

    * `:head_size` (required) - the dimensionality of each head

    * `:state_size` (required) - the size of the recurrent state per
      head channel

    * `:num_groups` - the number of groups that the heads are split
      into. Heads within a group share `B` and `C`. Defaults to `1`

    * `:conv_kernel_size` - the size of the short convolution applied
      before the recurrence. Defaults to `4`

    * `:chunk_size` - the number of positions processed in a single
      chunk. Defaults to `128`

    * `:activation` - the activation applied after the convolution.
      Defaults to `:silu`

    * `:time_step_min` - the lower bound that `dt` is clamped to.
      Defaults to `0.001`

    * `:layer_norm_epsilon` - the epsilon used by the output
      normalization. Defaults to `1.0e-5`

    * `:use_bias` - whether to use bias in the input and output
      projections. Defaults to `false`

    * `:use_conv_bias` - whether to use bias in the convolution.
      Defaults to `true`

    * `:name` - the prefix for layer names

  """
  def mixer(hidden_state, attention_mask, attention_cache, offset, opts) do
    validate_required_keys!(opts, [:hidden_size, :num_heads, :head_size, :state_size])

    opts =
      Keyword.validate!(opts, [
        :name,
        :hidden_size,
        :num_heads,
        :head_size,
        :state_size,
        num_groups: 1,
        conv_kernel_size: 4,
        chunk_size: 128,
        activation: :silu,
        time_step_min: 0.001,
        layer_norm_epsilon: 1.0e-5,
        use_bias: false,
        use_conv_bias: true,
        kernel_initializer: :glorot_uniform
      ])

    name = opts[:name]

    num_heads = opts[:num_heads]
    head_size = opts[:head_size]
    state_size = opts[:state_size]
    num_groups = opts[:num_groups]

    intermediate_size = num_heads * head_size
    group_size = num_groups * state_size
    conv_size = intermediate_size + 2 * group_size

    hidden_state = Layers.mask_padding(hidden_state, attention_mask, offset)

    projected =
      Axon.dense(hidden_state, intermediate_size + conv_size + num_heads,
        kernel_initializer: opts[:kernel_initializer],
        name: join(name, "input"),
        use_bias: opts[:use_bias]
      )

    gate = slice(projected, 0, intermediate_size)
    convolved = slice(projected, intermediate_size, conv_size)
    dt = slice(projected, intermediate_size + conv_size, num_heads)

    {full_convolved, attention_cache} =
      Layers.Decoder.cached_window_state(convolved, :convolution, attention_cache)

    convolved =
      convolved
      |> Layers.causal_depthwise_conv1d(full_convolved, attention_mask, offset,
        channels: conv_size,
        kernel_size: opts[:conv_kernel_size],
        use_bias: opts[:use_conv_bias],
        name: join(name, "convolution")
      )
      |> Layers.activation(opts[:activation])
      |> Layers.mask_padding(attention_mask, offset)

    input = slice(convolved, 0, intermediate_size)
    b = slice(convolved, intermediate_size, group_size)
    c = slice(convolved, intermediate_size + group_size, group_size)

    initial_state = Layers.Decoder.get_extra_state(attention_cache, :state)

    {output, final_state} =
      Axon.layer(
        &scan_impl/9,
        [
          input,
          dt,
          b,
          c,
          vector_parameter(hidden_state, num_heads, join(name, "a_log")),
          vector_parameter(hidden_state, num_heads, join(name, "d")),
          vector_parameter(hidden_state, num_heads, join(name, "time_step_bias")),
          Axon.optional(initial_state)
        ],
        num_heads: num_heads,
        head_size: head_size,
        state_size: state_size,
        num_groups: num_groups,
        chunk_size: opts[:chunk_size],
        time_step_min: opts[:time_step_min],
        op_name: :mamba_scan
      )
      |> Layers.unwrap_tuple(2)

    attention_cache = Layers.Decoder.put_extra_state(attention_cache, :state, final_state)

    output =
      output
      |> gated_rms_norm(gate,
        channels: intermediate_size,
        group_size: div(intermediate_size, num_groups),
        epsilon: opts[:layer_norm_epsilon],
        name: join(name, "output_norm")
      )
      |> Axon.dense(opts[:hidden_size],
        kernel_initializer: opts[:kernel_initializer],
        name: join(name, "output"),
        use_bias: opts[:use_bias]
      )

    {output, attention_cache}
  end

  defp slice(hidden_state, start, size) do
    Axon.nx(hidden_state, &Nx.slice_along_axis(&1, start, size, axis: -1))
  end

  defp vector_parameter(hidden_state, size, name) do
    parameter = Axon.param("value", fn _ -> {size} end, initializer: :zeros)

    Axon.layer(fn _hidden_state, value, _opts -> value end, [hidden_state, parameter],
      name: name,
      op_name: :mamba_parameter
    )
  end

  defp gated_rms_norm(hidden_state, gate, opts) do
    weight = Axon.param("weight", fn _, _ -> {opts[:channels]} end, initializer: :ones)

    Axon.layer(&gated_rms_norm_impl/4, [hidden_state, gate, weight],
      name: opts[:name],
      op_name: :gated_rms_norm,
      group_size: opts[:group_size],
      epsilon: opts[:epsilon]
    )
  end

  defp gated_rms_norm_impl(hidden_state, gate, weight, opts) do
    hidden_state = Nx.as_type(hidden_state, :f32)
    gate = Nx.as_type(gate, :f32)

    hidden_state = Nx.multiply(hidden_state, Axon.Activations.silu(gate))

    {batch_size, sequence_length, channels} = Nx.shape(hidden_state)
    group_size = opts[:group_size]

    grouped =
      Nx.reshape(
        hidden_state,
        {batch_size, sequence_length, div(channels, group_size), group_size}
      )

    variance = grouped |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)
    normalized = Nx.multiply(grouped, Nx.rsqrt(Nx.add(variance, opts[:epsilon])))

    normalized
    |> Nx.reshape({batch_size, sequence_length, channels})
    |> Nx.multiply(weight)
  end

  # Computes the chunked state-space recurrence. Note that all of the
  # shapes are known at compile time, so this is regular Elixir code
  # operating on tensor expressions.
  defp scan_impl(input, dt, b, c, a_log, d, time_step_bias, initial_state, opts) do
    num_heads = opts[:num_heads]
    head_size = opts[:head_size]
    state_size = opts[:state_size]
    num_groups = opts[:num_groups]
    time_step_min = opts[:time_step_min]

    {batch_size, sequence_length, _} = Nx.shape(input)

    input =
      input |> Nx.as_type(:f32) |> Nx.reshape({batch_size, sequence_length, num_heads, head_size})

    b = expand_groups(b, batch_size, sequence_length, num_groups, num_heads, state_size)
    c = expand_groups(c, batch_size, sequence_length, num_groups, num_heads, state_size)

    dt =
      dt
      |> Nx.as_type(:f32)
      |> Nx.add(time_step_bias)
      |> Axon.Activations.softplus()
      |> Nx.max(time_step_min)

    a = a_log |> Nx.as_type(:f32) |> Nx.exp() |> Nx.negate()

    # The number of positions computed in a single chunk. For short
    # sequences (in particular a single token during decoding) we use a
    # smaller chunk, so that we don't compute mostly padding
    chunk_size = min(opts[:chunk_size], sequence_length)
    padding = rem(chunk_size - rem(sequence_length, chunk_size), chunk_size)
    padded_length = sequence_length + padding
    num_chunks = div(padded_length, chunk_size)

    skip = Nx.multiply(Nx.new_axis(d, -1), pad_sequence(input, padding))

    # Discretize the input and the state transition
    scaled_input = Nx.multiply(input, Nx.new_axis(dt, -1))
    a = Nx.multiply(dt, a)

    chunks = fn tensor ->
      tensor
      |> pad_sequence(padding)
      |> Nx.reshape({batch_size, num_chunks, chunk_size, :auto, elem(Nx.shape(tensor), 3)})
      # {batch, chunks, chunk, heads, size} -> {batch, chunks, heads, chunk, size}
      |> Nx.transpose(axes: [0, 1, 3, 2, 4])
    end

    input_chunks = chunks.(scaled_input)
    b_chunks = chunks.(b)
    c_chunks = chunks.(c)

    # {batch, sequence, heads} -> {batch, heads, chunks, chunk}
    a_chunks =
      a
      |> Nx.pad(0.0, [{0, 0, 0}, {0, padding, 0}, {0, 0, 0}])
      |> Nx.reshape({batch_size, num_chunks, chunk_size, num_heads})
      |> Nx.transpose(axes: [0, 3, 1, 2])

    a_cumulative = Nx.cumulative_sum(a_chunks, axis: -1)

    # 1. The contribution of the positions within the same chunk. This
    # is the quadratic, attention-like form, where the decay factors
    # play the role of a causal mask
    decay = a_chunks |> segment_sum() |> Nx.exp() |> Nx.transpose(axes: [0, 2, 1, 3, 4])

    scores =
      c_chunks
      |> Nx.dot([4], [0, 1, 2], b_chunks, [4], [0, 1, 2])
      |> Nx.multiply(decay)

    within_chunk = Nx.dot(scores, [4], [0, 1, 2], input_chunks, [3], [0, 1, 2])

    # 2. The state at the end of each chunk, ignoring the preceding chunks
    chunk_decay =
      a_cumulative
      |> Nx.slice_along_axis(chunk_size - 1, 1, axis: -1)
      |> Nx.subtract(a_cumulative)
      |> Nx.exp()
      # {batch, heads, chunks, chunk} -> {batch, chunks, heads, chunk}
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.new_axis(-1)

    chunk_states =
      Nx.dot(input_chunks, [3], [0, 1, 2], Nx.multiply(b_chunks, chunk_decay), [3], [0, 1, 2])

    # 3. The recurrence over chunks, which carries the state across
    # chunk boundaries
    initial_state =
      case initial_state do
        %Axon.None{} -> Nx.broadcast(0.0, {batch_size, 1, num_heads, head_size, state_size})
        state -> state |> Nx.as_type(:f32) |> Nx.new_axis(1)
      end

    states =
      Nx.concatenate([initial_state, chunk_states], axis: 1)
      # {batch, chunks, heads, head_size, state} -> {batch, heads, chunks, head_size, state}
      |> Nx.transpose(axes: [0, 2, 1, 3, 4])

    boundary_decay =
      a_cumulative
      |> Nx.slice_along_axis(chunk_size - 1, 1, axis: -1)
      |> Nx.squeeze(axes: [-1])
      |> Nx.pad(0.0, [{0, 0, 0}, {0, 0, 0}, {1, 0, 0}])
      |> segment_sum()
      |> Nx.exp()

    states = Nx.dot(boundary_decay, [3], [0, 1], states, [2], [0, 1])

    final_state = states[[.., .., num_chunks]]

    # 4. The contribution of the state carried into each chunk
    across_chunks =
      c_chunks
      |> Nx.dot(
        [4],
        [0, 1, 2],
        states[[.., .., 0..(num_chunks - 1)//1]] |> Nx.transpose(axes: [0, 2, 1, 3, 4]),
        [4],
        [0, 1, 2]
      )
      |> Nx.multiply(
        a_cumulative
        |> Nx.exp()
        |> Nx.transpose(axes: [0, 2, 1, 3])
        |> Nx.new_axis(-1)
      )

    output =
      within_chunk
      |> Nx.add(across_chunks)
      # {batch, chunks, heads, chunk, head_size} -> {batch, chunks, chunk, heads, head_size}
      |> Nx.transpose(axes: [0, 1, 3, 2, 4])
      |> Nx.reshape({batch_size, padded_length, num_heads, head_size})
      |> Nx.add(skip)
      |> Nx.slice_along_axis(0, sequence_length, axis: 1)
      |> Nx.reshape({batch_size, sequence_length, num_heads * head_size})

    {output, final_state}
  end

  defp pad_sequence(tensor, padding) do
    config = [{0, 0, 0}, {0, padding, 0}] ++ List.duplicate({0, 0, 0}, Nx.rank(tensor) - 2)
    Nx.pad(tensor, 0.0, config)
  end

  # Repeats the group states, so that every head has its own copy
  defp expand_groups(tensor, batch_size, sequence_length, num_groups, num_heads, state_size) do
    repeats = div(num_heads, num_groups)

    tensor
    |> Nx.as_type(:f32)
    |> Nx.reshape({batch_size, sequence_length, num_groups, 1, state_size})
    |> Nx.broadcast({batch_size, sequence_length, num_groups, repeats, state_size})
    |> Nx.reshape({batch_size, sequence_length, num_heads, state_size})
  end

  # Given `x` of shape `{..., n}`, returns a tensor of shape
  # `{..., n, n}`, where the entry at `[i, j]` is the sum of `x[j + 1..i]`
  # for `j <= i` and negative infinity otherwise. Accumulating the decay
  # factors this way, rather than dividing cumulative sums, is what keeps
  # the chunked form numerically stable.
  defp segment_sum(tensor) do
    size = Nx.axis_size(tensor, -1)

    rows = Nx.iota({size, size}, axis: 0)
    columns = Nx.iota({size, size}, axis: 1)

    summed =
      tensor
      |> Nx.new_axis(-1)
      |> Nx.multiply(Nx.as_type(Nx.greater(rows, columns), Nx.type(tensor)))
      |> Nx.cumulative_sum(axis: -2)

    mask = Nx.broadcast(Nx.greater_equal(rows, columns), Nx.shape(summed))

    Nx.select(mask, summed, Nx.Constants.neg_infinity())
  end

  defp validate_required_keys!(opts, keys) do
    missing = keys -- Keyword.keys(opts)

    if missing != [] do
      raise ArgumentError, "missing required options: #{inspect(missing)}"
    end
  end
end
