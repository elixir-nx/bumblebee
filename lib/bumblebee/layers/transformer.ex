defmodule Bumblebee.Layers.Transformer do
  @moduledoc false

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @doc """
  Adds a stack of transformer blocks to the network.

  This function handles the parts shared by most transformer models:
  the attention cache, per-block head masks and collecting outputs.
  The block itself is defined by `fun`.

  `fun` receives the hidden state and a `block` map with the following
  keys:

    * `:index` - the zero-based block index

    * `:name` - the prefix for block layer names

    * `:attention_mask` - the attention mask, combined with the cache
      mask during iterative decoding

    * `:attention_head_mask` - the head mask for this block

    * `:cross_attention_head_mask` - the cross-attention head mask for
      this block

    * `:offset` - offset in the input sequence during iterative decoding

    * `:self_attention_cache`, `:cross_attention_cache` - attention
      caches for this block

  `fun` must return a map with `:hidden_state`. It may also include
  `:attention`, `:cross_attention`, `:self_attention_cache` and
  `:cross_attention_cache`.

  ## Options

    * `:num_blocks` (required) - the number of blocks

    * `:attention_mask` - a mask indicating which positions to attend to

    * `:attention_head_mask` - a mask to nullify selected attention heads,
      with a leading axis for blocks

    * `:cross_attention_head_mask` - same as `:attention_head_mask`, but
      for cross-attention

    * `:cache` - cache for iterative decoding

    * `:name` - the prefix for layer names

  """
  def blocks(hidden_state, opts, fun) do
    opts =
      Keyword.validate!(opts, [
        :name,
        :num_blocks,
        attention_mask: Layers.none(),
        attention_head_mask: Layers.none(),
        cross_attention_head_mask: Layers.none(),
        cache: Layers.none()
      ])

    {attention_mask, cache} =
      Layers.Decoder.cached_attention_mask(opts[:attention_mask], opts[:cache])

    offset = Layers.Decoder.get_cache_offset(cache)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cross_attentions: Axon.container({}),
      cache: cache
    }

    outputs =
      for index <- 0..(opts[:num_blocks] - 1), reduce: state do
        state ->
          block_cache = Layers.Decoder.get_block_cache(state.cache, index)

          {self_attention_cache, cross_attention_cache} =
            Layers.Decoder.get_attention_caches(block_cache)

          block = %{
            index: index,
            name: join(opts[:name], index),
            attention_mask: attention_mask,
            attention_head_mask: Axon.nx(opts[:attention_head_mask], & &1[index]),
            cross_attention_head_mask: Axon.nx(opts[:cross_attention_head_mask], & &1[index]),
            offset: offset,
            self_attention_cache: self_attention_cache,
            cross_attention_cache: cross_attention_cache
          }

          result =
            Map.merge(
              %{
                attention: Layers.none(),
                cross_attention: Layers.none(),
                self_attention_cache: self_attention_cache,
                cross_attention_cache: cross_attention_cache
              },
              fun.(state.hidden_state, block)
            )

          block_cache =
            Layers.Decoder.put_attention_caches(
              block_cache,
              result.self_attention_cache,
              result.cross_attention_cache
            )

          %{
            hidden_state: result.hidden_state,
            hidden_states: Layers.append(state.hidden_states, result.hidden_state),
            attentions: Layers.append(state.attentions, result.attention),
            cross_attentions: Layers.append(state.cross_attentions, result.cross_attention),
            cache: Layers.Decoder.put_block_cache(state.cache, index, block_cache)
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  @doc """
  Adds a self-attention layer to a block in `blocks/3`.

  Takes the mask, cache and offset from `block`. See
  `multi_head_attention/4` for options.

  Returns `{output, attention_weights, attention_cache}`.
  """
  def self_attention(hidden_state, block, opts) do
    {output, weights, cache, _attention_relative_bias} =
      multi_head_attention(
        hidden_state,
        hidden_state,
        hidden_state,
        [
          attention_mask: block.attention_mask,
          attention_head_mask: block.attention_head_mask,
          attention_cache: block.self_attention_cache,
          offset: block.offset
        ] ++ opts
      )

    {output, weights, cache}
  end

  @doc """
  Adds a cross-attention layer to a block in `blocks/3`.

  Takes the head mask, cache and offset from `block`. The cross
  attention mask should be given as `:attention_mask`. See
  `multi_head_attention/4` for other options.

  Returns `{output, attention_weights, attention_cache}`.
  """
  def cross_attention(hidden_state, cross_hidden_state, block, opts) do
    {output, weights, cache, _attention_relative_bias} =
      multi_head_attention(
        hidden_state,
        cross_hidden_state,
        cross_hidden_state,
        [
          attention_head_mask: block.cross_attention_head_mask,
          attention_cache: block.cross_attention_cache,
          offset: block.offset
        ] ++ opts
      )

    {output, weights, cache}
  end

  @doc """
  Adds a feed-forward network with two dense layers and an activation
  in-between.

  ## Options

    * `:activation` - the activation. Defaults to `:gelu`

    * `:dropout_rate` - the dropout rate at the end. Defaults to `0.0`

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`

    * `:name` - the prefix for layer names

  """
  def basic_ffn(x, intermediate_size, output_size, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        activation: :gelu,
        dropout_rate: 0.0,
        kernel_initializer: :glorot_uniform
      ])

    name = opts[:name]

    x
    |> Axon.dense(intermediate_size,
      kernel_initializer: opts[:kernel_initializer],
      name: join(name, "intermediate")
    )
    |> Layers.activation(opts[:activation])
    |> Axon.dense(output_size,
      kernel_initializer: opts[:kernel_initializer],
      name: join(name, "output")
    )
    |> Axon.dropout(rate: opts[:dropout_rate])
  end

  @doc """
  Adds a gated feed-forward network, as used in most recent LLMs.

  ## Options

    * `:activation` - the activation applied to the gate. Defaults
      to `:silu`

    * `:use_bias` - whether to use bias in the dense layers. Defaults
      to `false`

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`

    * `:name` - the prefix for layer names

  """
  def gated_ffn(x, intermediate_size, output_size, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        activation: :silu,
        use_bias: false,
        kernel_initializer: :glorot_uniform
      ])

    name = opts[:name]

    dense = fn x, size, name ->
      Axon.dense(x, size,
        use_bias: opts[:use_bias],
        kernel_initializer: opts[:kernel_initializer],
        name: name
      )
    end

    intermediate = dense.(x, intermediate_size, join(name, "intermediate"))
    gate = dense.(x, intermediate_size, join(name, "gate"))

    intermediate
    |> Axon.multiply(Layers.activation(gate, opts[:activation]))
    |> dense.(output_size, join(name, "output"))
  end

  @doc """
  Adds a multi-head attention block to the network.

  When `query`, `key` and `value` are the same, this is self-attention.
  When `query` comes from the decoder, while `key` and `value` come from
  the encoder, this is cross-attention.

  Returns the tuple `{attention_output, attention_weights, attention_cache}`.

  ## Options

    * `:num_heads` (required) - the number of attention heads

    * `:hidden_size` (required) - the dimensionality of query/key/value
      projections

    * `:attention_mask` - a mask indicating which positions to attend to

    * `:attention_head_mask` - a mask to nullify selected attention heads

    * `:attention_relative_bias` - configuration of relative bias. If set,
      will apply relative attention bias with the given options. Valid
      options are:

        * `:num_buckets` (required) - number of relative attention buckets

        * `:max_distance` (required) - maximum distance of the relative attention
          bias

        * `:bidirectional` (required) - whether to apply the relative attention
          bias bidirectionally

      Alternatively an `Axon` node may be given with the computed bias.

    * `:attention_cache` - cache with accumulated key/values useful for
      iterative decoding

    * `:offset` - offset in the input sequence during iterative decoding

    * `:causal` - whether to apply causal attention mask, so that tokens
      are attended to only in a single direction. Defaults to `false`

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`

    * `:dropout_rate` - the dropout rate for attention weights dropout.
      Defaults to `0.0`

    * `:attention_head_size` - the projection size for key, value,
      and query states per-head. Defaults to `div(hidden_size, num_attention_heads)`

    * `:query_use_bias` - whether to use bias in the query projection.
      Defaults to `true`

    * `:key_use_bias` - whether to use bias in the key projection.
      Defaults to `true`

    * `:value_use_bias` - whether to use bias in the value projection.
      Defaults to `true`

    * `:output_use_bias` - whether to use bias in the output projection.
      Defaults to `true`

    * `:attention_window_size` - when set, enables sliding window attention.
      Should be a `{left, right}` tuple with window size on each side

    * `:attention_scale` - the scaling factor applied to the attention weights.
      Defaults to $\frac{1}{\sqrt{d}}$

    * `:rotary_embedding` - configuration of rotary embedding. If set,
      will apply rotary position embedding with the given options. Valid
      options are:

        * `:position_ids` (required) - input position ids used for the
          embedding

        * `:max_positions` - the maximum number of distinct positions

    * `:query_norm` - a function that applies normalization to the query
      projection before rotary embedding. The function should accept two
      arguments: the input and a name for the layer. Defaults to `nil`

    * `:key_norm` - a function that applies normalization to the key
      projection before rotary embedding. The function should accept two
      arguments: the input and a name for the layer. Defaults to `nil`

    * `:name` - the prefix for layer names

  ## References

    * [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Figure 2 (right)

  """
  def multi_head_attention(query, key, value, opts) do
    validate_required_keys!(opts, [:num_heads, :hidden_size])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_heads,
        :hidden_size,
        :num_key_value_heads,
        attention_mask: Layers.none(),
        attention_head_mask: Layers.none(),
        attention_relative_bias: Layers.none(),
        attention_cache: Layers.none(),
        offset: Layers.none(),
        causal: false,
        attention_window_size: nil,
        attention_scale: nil,
        kernel_initializer: :glorot_uniform,
        dropout_rate: 0.0,
        attention_head_size: nil,
        query_use_bias: true,
        key_use_bias: true,
        value_use_bias: true,
        output_use_bias: true,
        rotary_embedding: nil,
        query_norm: nil,
        key_norm: nil
      ])

    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    attention_cache = opts[:attention_cache]
    offset = opts[:offset]

    name = opts[:name]
    num_heads = opts[:num_heads]
    num_key_value_heads = opts[:num_key_value_heads] || num_heads
    hidden_size = opts[:hidden_size]
    kernel_initializer = opts[:kernel_initializer]
    causal = opts[:causal]
    attention_window_size = opts[:attention_window_size]
    attention_scale = opts[:attention_scale]
    dropout_rate = opts[:dropout_rate]
    rotary_embedding = opts[:rotary_embedding]
    query_norm = opts[:query_norm]
    key_norm = opts[:key_norm]

    query_use_bias = opts[:query_use_bias]
    key_use_bias = opts[:key_use_bias]
    value_use_bias = opts[:value_use_bias]
    output_use_bias = opts[:output_use_bias]

    attention_relative_bias = opts[:attention_relative_bias]

    attention_head_size = opts[:attention_head_size] || div(hidden_size, num_heads)

    project = fn input, num_heads, use_bias, suffix ->
      project_heads(input, num_heads, attention_head_size,
        use_bias: use_bias,
        kernel_initializer: kernel_initializer,
        name: join(name, suffix)
      )
    end

    query = project.(query, num_heads, query_use_bias, "query")
    key = project.(key, num_key_value_heads, key_use_bias, "key")
    value = project.(value, num_key_value_heads, value_use_bias, "value")

    query = if query_norm, do: query_norm.(query, join(name, "query_norm")), else: query
    key = if key_norm, do: key_norm.(key, join(name, "key_norm")), else: key

    {query, key} =
      if rotary_embedding do
        rotary_embedding(query, key, attention_mask, attention_head_size, rotary_embedding,
          name: join(name, "rotary_embedding")
        )
      else
        {query, key}
      end

    num_key_value_groups = div(num_heads, num_key_value_heads)
    key = repeat_heads(key, num_key_value_groups)
    value = repeat_heads(value, num_key_value_groups)

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, attention_cache, offset)

    attention_relative_bias =
      case attention_relative_bias do
        %Axon{} ->
          attention_relative_bias

        bias_opts when is_list(bias_opts) ->
          validate_required_keys!(bias_opts, [:num_buckets, :max_distance, :bidirectional])
          bias_opts = Keyword.validate!(bias_opts, [:num_buckets, :max_distance, :bidirectional])

          Layers.relative_attention_bias(query, key, attention_cache, offset,
            num_buckets: bias_opts[:num_buckets],
            max_distance: bias_opts[:max_distance],
            bidirectional: bias_opts[:bidirectional],
            num_heads: num_heads,
            name: join(name, "relative_attention_bias")
          )
      end

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        attention_mask,
        attention_head_mask,
        attention_relative_bias,
        offset,
        scale: attention_scale,
        causal: causal,
        window_size: attention_window_size,
        dropout_rate: dropout_rate
      )

    attention_output =
      output_projection(attention_output, hidden_size,
        use_bias: output_use_bias,
        kernel_initializer: kernel_initializer,
        name: join(name, "output")
      )

    {attention_output, attention_weights, attention_cache, attention_relative_bias}
  end

  @doc """
  Projects the input to attention heads.

  Returns a node with shape `{batch_size, sequence_length, num_heads, head_size}`.

  ## Options

    * `:use_bias` - whether to use bias in the projection. Defaults
      to `true`

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`

    * `:name` - the layer name

  """
  def project_heads(input, num_heads, head_size, opts) do
    opts = Keyword.validate!(opts, [:name, use_bias: true, kernel_initializer: :glorot_uniform])

    input
    |> Axon.dense(num_heads * head_size,
      use_bias: opts[:use_bias],
      kernel_initializer: opts[:kernel_initializer],
      name: opts[:name]
    )
    |> Layers.split_heads(num_heads)
  end

  @doc """
  Merges attention heads and projects them to `hidden_size`.

  Accepts the same options as `project_heads/4`.
  """
  def output_projection(attention_output, hidden_size, opts) do
    opts = Keyword.validate!(opts, [:name, use_bias: true, kernel_initializer: :glorot_uniform])

    attention_output
    |> Layers.flatten_trailing()
    |> Axon.dense(hidden_size,
      use_bias: opts[:use_bias],
      kernel_initializer: opts[:kernel_initializer],
      name: opts[:name]
    )
  end

  @doc """
  Applies rotary embedding to query and key heads.

  ## Rotary options

    * `:position_ids` (required) - input position ids used for the
      embedding

    * `:max_positions` - the maximum number of distinct positions

    * `:base` - base for computing rotary embedding frequency. Defaults
      to `10_000`

    * `:scaling_strategy` - see `Bumblebee.Layers.rotary_embedding/6`

    * `:percentage` - percentage of head dimensions to apply rotary
      embedding to. Defaults to `1.0`

  ## Options

    * `:name` - the prefix for layer names

  """
  def rotary_embedding(query, key, attention_mask, head_size, rotary_opts, opts) do
    validate_required_keys!(rotary_opts, [:position_ids])

    rotary_opts =
      Keyword.validate!(rotary_opts, [
        :position_ids,
        :max_positions,
        :scaling_strategy,
        base: 10_000,
        percentage: 1.0
      ])

    {position_ids, rotary_opts} = Keyword.pop(rotary_opts, :position_ids)
    {percentage, rotary_opts} = Keyword.pop(rotary_opts, :percentage)

    size = trunc(head_size * percentage)

    rotary_opts = [name: opts[:name]] ++ rotary_opts

    if size == head_size do
      Layers.rotary_embedding(query, key, position_ids, attention_mask, size, rotary_opts)
    else
      query_rotary = Axon.nx(query, & &1[[.., .., .., 0..(size - 1)//1]])
      query_pass = Axon.nx(query, & &1[[.., .., .., size..-1//1]])

      key_rotary = Axon.nx(key, & &1[[.., .., .., 0..(size - 1)//1]])
      key_pass = Axon.nx(key, & &1[[.., .., .., size..-1//1]])

      {query_rotary, key_rotary} =
        Layers.rotary_embedding(
          query_rotary,
          key_rotary,
          position_ids,
          attention_mask,
          size,
          rotary_opts
        )

      {Axon.concatenate([query_rotary, query_pass], axis: -1),
       Axon.concatenate([key_rotary, key_pass], axis: -1)}
    end
  end

  @doc """
  Repeats key or value heads to match the number of query heads, as
  in grouped-query attention.
  """
  def repeat_heads(state, 1), do: state

  def repeat_heads(state, times) do
    Layers.repeat_interleave(state, times, axis: 2)
  end

  defp validate_required_keys!(opts, keys) do
    case keys -- Keyword.keys(opts) do
      [] -> :ok
      missing -> raise ArgumentError, "missing required options: #{inspect(missing)}"
    end
  end
end
