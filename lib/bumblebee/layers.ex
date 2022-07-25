defmodule Bumblebee.Layers do
  @moduledoc false

  import Nx.Defn

  @unsupported_activations [:gelu_new]

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
  Implements the GeLU new activation from huggingface/transformers.
  """
  defn gelu_new(input, _opts \\ []) do
    0.5 * input *
      (1.0 + Nx.tanh(Nx.sqrt(2.0 / @pi) * (input + 0.044715 * Nx.power(input, 3.0))))
  end

  @doc """
  Expands an attention mask of shape `{batch_size, seq_length}` to
  a full mask.
  """
  def expand_attention_mask(attention_mask) do
    Axon.nx(attention_mask, fn attention_mask ->
      attention_mask
      |> Nx.new_axis(-2)
      |> Nx.new_axis(-2)
    end)
  end

  @doc """
  Converts attention mask to bias.
  """
  def attention_bias(attention_mask) do
    Axon.nx(attention_mask, fn attention_mask ->
      Nx.select(Nx.greater(attention_mask, 0), 0, -1.0e10)
    end)
  end

  @doc """
  Computes attention weights.
  """
  def attention_weights(query, key, bias) do
    Axon.layer(&attention_weights_impl/4, [query, key, bias])
  end

  defnp attention_weights_impl(query, key, bias, _opts \\ []) do
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
  def attention_output(attention_weights, value) do
    Axon.layer(&attention_output_impl/3, [attention_weights, value])
  end

  defnp attention_output_impl(attention_weights, value, _opts \\ []) do
    value = Nx.transpose(value, axes: [0, 2, 1, 3])
    out = Nx.dot(attention_weights, [3], [0, 1], value, [2], [0, 1])
    Nx.transpose(out, axes: [0, 2, 1, 3])
  end

  @doc """
  Applies head mask to the given attention weights.

  This layer expects computed attention weights and an optional mask.
  If the mask is not specified, it will skip masking altogether.
  """
  def apply_layer_head_mask(attention_weights, layer_head_mask) do
    if_present layer_head_mask do
      Axon.layer(
        fn attention_weights, layer_head_mask, _ ->
          layer_head_mask = Nx.reshape(layer_head_mask, {1, :auto, 1, 1})
          Nx.multiply(attention_weights, layer_head_mask)
        end,
        [attention_weights, layer_head_mask]
      )
    else
      attention_weights
    end
  end

  @doc """
  Adds a dense layer to the network.

  The kernel parameter is transposed with respect to `Axon.dense/3`.

  ## Options

    * `:name` - layer name

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`

  """
  def dense_transposed(%Axon{} = x, units, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, kernel_initializer: :glorot_uniform])

    kernel_shape = fn input_shape ->
      kernel_shape = Axon.Shape.dense_kernel(input_shape, units)

      # We expect a transposed kernel
      kernel_shape
      |> Tuple.to_list()
      |> Enum.reverse()
      |> List.to_tuple()
    end

    kernel = Axon.param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    op = fn x, kernel, _opts ->
      Nx.dot(x, [-1], kernel, [1])
    end

    Axon.layer(op, [x, kernel], name: opts[:name], op_name: :dense_transposed)
  end

  @doc """
  Adds a scaling layer to the network.

  The scaling layer scales inputs by a learned scale parameter.

  ## Options

    * `:name` - layer name

    * `:scale_name` - scale parameter name

    * `:scale_init_value` - initial value of scale parameter

  """
  def scale(%Axon{} = x, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        scale_name: "scale",
        scale_init_value: 1.0e-6,
        channel_index: 1
      ])

    name = opts[:name]
    scale_name = opts[:scale_name]
    scale_init_value = opts[:scale_init_value]
    channel_index = opts[:channel_index]

    scale_shape = fn input_shape ->
      out_channels = elem(input_shape, channel_index)
      {out_channels}
    end

    scale_param =
      Axon.param(scale_name, scale_shape,
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
  Adds a drop-path layer to the network for stochastic depth.

  ## Options

    * `:name` - layer name

    * `:rate` - drop path rate

  """
  def drop_path(%Axon{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, rate: 0.0])

    if opts[:rate] > 0.0 do
      Axon.layer(&drop_path_impl/2, [input],
        name: opts[:name],
        op_name: :drop_path,
        rate: opts[:rate]
      )
    else
      input
    end
  end

  defnp drop_path_impl(x, opts \\ []) do
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
  def apply_vision_patch_mask(%Axon{} = embeds, patch_mask, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])
    name = opts[:name]

    mask_token_shape = fn embeds_shape, _ ->
      hidden_size = elem(embeds_shape, 2)
      {1, 1, hidden_size}
    end

    mask_token = Axon.param("mask_token", mask_token_shape, initializer: :zeros)

    if_present patch_mask do
      Axon.layer(
        fn embeds, patch_mask, mask_tokens, _opts ->
          hidden_size = Nx.axis_size(embeds, 2)
          batch_size = Nx.axis_size(embeds, 0)
          seq_len = Nx.axis_size(embeds, 1)
          mask_tokens = Nx.broadcast(mask_tokens, {batch_size, seq_len, hidden_size})
          mask = patch_mask |> Nx.new_axis(-1) |> Nx.broadcast({batch_size, seq_len, hidden_size})
          Nx.select(mask, mask_tokens, embeds)
        end,
        [embeds, patch_mask, mask_token],
        name: name,
        op_name: :apply_patch_mask
      )
    else
      embeds
    end
  end

  @doc """
  Splits the hidden dimension into the given number of attention heads.

  In other words, the input with shape `{batch_size, seq_length, hidden_size}`
  is reshaped to `{batch_size, seq_length, num_heads, *}`.
  """
  def split_heads(states, num_heads) do
    Axon.nx(states, fn states ->
      batch_size = Nx.axis_size(states, 0)
      seq_length = Nx.axis_size(states, 1)
      new_shape = {batch_size, seq_length, num_heads, :auto}
      Nx.reshape(states, new_shape)
    end)
  end

  @doc """
  Splits the input node with shape `{bach_size, seq_length, 2}` into
  two nodes with shape `{batch_size, seq_length}`.
  """
  def split_pair(%Axon{} = x) do
    left = Axon.nx(x, & &1[[0..-1//1, 0..-1//1, 0]])
    right = Axon.nx(x, & &1[[0..-1//1, 0..-1//1, 1]])
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

  Rearranges elements in a tensor of shape `{*, C × r^2, H, W}` to a
  tensor of shape `{*, C, H × r, W × r}`, where r is an upscale factor.

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

  defnp pixel_shuffle_impl(input, opts \\ []) do
    opts = keyword!(opts, [:upscale_factor, mode: :inference])

    transform({input, opts[:upscale_factor]}, fn {input, upscale_factor} ->
      {batch, [channels, height, width]} =
        input
        |> Nx.shape()
        |> Tuple.to_list()
        |> Enum.split(-3)

      out_channels = div(channels, upscale_factor * upscale_factor)
      out_height = height * upscale_factor
      out_width = width * upscale_factor

      x =
        Nx.reshape(
          input,
          List.to_tuple(batch ++ [out_channels, upscale_factor, upscale_factor, height, width])
        )

      {batch_axes, [out_channels_axis, upscale_axis1, upscale_axis2, height_axis, width_axis]} =
        x
        |> Nx.axes()
        |> Enum.split(-5)

      x
      |> Nx.transpose(
        axes:
          batch_axes ++ [out_channels_axis, height_axis, upscale_axis1, width_axis, upscale_axis2]
      )
      |> Nx.reshape(List.to_tuple(batch ++ [out_channels, out_height, out_width]))
    end)
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
        Bumblebee.Layers.conditional_if encoder_hidden_state do
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
  Returns a container layer if `condition` is truthy, otherwise returns
  a none layer.
  """
  def maybe_container(container, condition) do
    if condition do
      Axon.container(container)
    else
      none()
    end
  end

  @doc """
  Performs `Tuple.append/1` on node results.
  """
  def append(%Axon{} = tuple, %Axon{} = x) do
    Axon.layer(fn tuple, x, _ -> Tuple.append(tuple, x) end, [tuple, x], op_name: :append)
  end

  @doc """
  Builds an `Axon` container with the given outputs.

  All values are wrapped with `Axon.optional/2`, so if any of them is
  missing, it gets returned as `%Axon.None{}`.
  """
  @spec output(map()) :: Axon.t()
  def output(outputs) do
    outputs
    |> Map.new(fn
      {key, %Axon{} = val} -> {key, Axon.optional(val)}
      {key, val} -> {key, val}
    end)
    |> Axon.container()
  end

  @doc """
  Computes a 1-full mask matching the first two dimensions of `input`
  (batch size and sequence length).
  """
  def default_attention_mask(%Axon{} = input) do
    Axon.nx(input, fn input ->
      batch_size = Nx.axis_size(input, 0)
      seq_length = Nx.axis_size(input, 1)
      Nx.broadcast(1, {batch_size, seq_length})
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
      seq_length = Nx.axis_size(input, 1)
      Nx.iota({batch_size, seq_length}, axis: -1) |> Nx.add(offset)
    end)
  end

  @doc """
  Computes 0-full mask matching the first two dimensions of `input`
  (batch size and sequence length).
  """
  def default_token_type_ids(%Axon{} = input) do
    Axon.nx(input, fn input ->
      batch_size = Nx.axis_size(input, 0)
      seq_length = Nx.axis_size(input, 1)
      Nx.broadcast(0, {batch_size, seq_length})
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
      seq_length = Nx.axis_size(input_ids, 1)
      start_ids = Nx.broadcast(decoder_start_token_id, {batch_size, 1})

      if seq_length == 1 do
        start_ids
      else
        Nx.concatenate([start_ids, input_ids[[0..-1//1, 0..-2//1]]], axis: 1)
      end
    end)
  end
end
