defmodule Bumblebee.Layers.Transformer do
  @moduledoc false

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @doc """
  Adds a list of transformer block to the network.

  Adds transformer blocks (see `block/2`), while also accumulating
  caches and outputs. This is a convenience function, since this
  chunk is oftentimes the same across transformer models.

  ## Options

    * `:num_blocks` (required) - the number of consecutive transformer
      blocks to add

    * `:share_attention_relative_bias` - when attention relative bias
      is configured, this option controls whether the bias from the
      first block is used for all other blocks. Defaults to `false`

    * `:output_hidden_states` - when `true`, the output includes a
      tuple with intermediate hidden states from each transformer
      block. Defaults to `false`

    * `:output_attentions` - when `true`, the output includes a tuple
      with attention weights from each transformer block. Defaults
      to `false`

    * `:name` - the prefix for layer names

  For all other options (including required options) see `block/2`.
  Note that `:attention_head_mask`, `:cross_attention_head_mask` should
  have an additional leading axis corresponding to the number of blocks,
  since each block gets its own mask. Similarly `:cache` is a tuple of
  block caches.
  """
  def blocks(hidden_state, opts) do
    validate_required_keys!(opts, [:num_blocks, :num_attention_heads, :hidden_size, :ffn])

    block_opts_keys = [
      :num_attention_heads,
      :causal?,
      :hidden_size,
      :ffn,
      :kernel_initializer,
      :attention_head_size,
      :dropout_rate,
      :attention_dropout_rate,
      :query_use_bias,
      :key_use_bias,
      :value_use_bias,
      :output_use_bias,
      :layer_norm,
      :norm_placement,
      :output_shortcut,
      :scale_query?,
      :use_rotary_embedding?,
      :position_ids
    ]

    opts =
      Keyword.validate!(
        opts,
        block_opts_keys ++
          [
            :name,
            :num_blocks,
            attention_mask: Layers.none(),
            attention_head_mask: Layers.none(),
            attention_relative_bias: nil,
            share_attention_relative_bias: false,
            cross_hidden_state: nil,
            cross_attention_mask: Layers.none(),
            cross_attention_head_mask: Layers.none(),
            cache: Layers.none(),
            output_hidden_states: false,
            output_attentions: false
          ]
      )

    name = opts[:name]
    num_blocks = opts[:num_blocks]
    output_hidden_states = opts[:output_hidden_states]
    output_attentions = opts[:output_attentions]

    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    cross_hidden_state = opts[:cross_hidden_state]
    cross_attention_mask = opts[:cross_attention_mask]
    cross_attention_head_mask = opts[:cross_attention_head_mask]
    cache = opts[:cache]

    block_opts = Keyword.take(opts, block_opts_keys)

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, output_hidden_states),
      attentions: Layers.maybe_container({}, output_attentions),
      cross_attentions: Layers.maybe_container({}, output_attentions),
      cache: cache,
      attention_relative_bias: Layers.none()
    }

    outputs =
      for idx <- 0..(num_blocks - 1), reduce: state do
        state ->
          block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])
          block_cross_attention_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])
          block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

          attention_relative_bias =
            if opts[:share_attention_relative_bias] and idx > 0 do
              state.attention_relative_bias
            else
              opts[:attention_relative_bias] || Layers.none()
            end

          {hidden_state, attention, cross_attention, block_cache, attention_relative_bias} =
            block(
              state.hidden_state,
              [
                attention_mask: attention_mask,
                attention_head_mask: block_attention_head_mask,
                attention_relative_bias: attention_relative_bias,
                cross_hidden_state: cross_hidden_state,
                cross_attention_mask: cross_attention_mask,
                cross_attention_head_mask: block_cross_attention_head_mask,
                block_cache: block_cache,
                offset: offset,
                name: join(name, idx)
              ] ++ block_opts
            )

          cache = Layers.Decoder.put_block_cache(state.cache, idx, block_cache)

          %{
            hidden_state: hidden_state,
            hidden_states: Layers.append(state.hidden_states, hidden_state),
            attentions: Layers.append(state.attentions, attention),
            cross_attentions: Layers.append(state.cross_attentions, cross_attention),
            attention_relative_bias: attention_relative_bias,
            cache: cache
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  @doc """
  Adds a transformer block to the network

  Depending on the configuration, this can implement both the encoder
  and the decoder block.

  A transformer block consists of self-attention, shortcut connection,
  normalization and dropout layers. It may also include a cross-attention
  block.

  ## Options

    * `:num_attention_heads` (required) - the number of attention heads

    * `:hidden_size` (required) - the dimensionality of the attention
      block and the last layer in the output feed-forward network

    * `:ffn` (required) - configuration of the feed-forward network at
      the end of the transformer block. By default the network has two
      dense layers with an activation in-between and is configured with
      the following options:

      * `:intermediate_size` (required) - the dimensionality of the
        first dense layer

      * `:activation` - the activation used in-between the layers.
        Defaults to `:gelu`

      Alternatively a custom 2-arity function may be given. The function
      should add FFN nodes to the given Axon node. The function also
      receives layer name prefix as the second argument.

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

    * `:cross_hidden_state` - the second input for the the cross-attention
      block. The cross-attention block is added only when this is specified.
      Defaults to `nil`

    * `:cross_attention_mask` - a mask indicating which positions to
      attend to in the cross-attention block

    * `:cross_attention_head_mask` - a mask to nullify selected attention
      heads in the cross-attention block

    * `:block_cache` - cache for all attention blocks in this block,
      used in iterative decoding

    * `:offset` - offset in the input sequence during iterative decoding

    * `:causal?` - whether the self-attention block should be causal.
      Defaults to `false`

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`

    * `:attention_head_size` - the projection size for key, value,
      and query states per-head. Defaults to `div(hidden_size, num_attention_heads)`

    * `:dropout_rate` - the dropout rate for dropout layers. Defaults
      to `0.0`

    * `:attention_dropout_rate` - the dropout rate for attention weights
      dropout. Defaults to `0.0`

    * `:query_use_bias` - whether to use bias in the query projection.
      Defaults to `true`

    * `:key_use_bias` - whether to use bias in the key projection.
      Defaults to `true`

    * `:value_use_bias` - whether to use bias in the value projection.
      Defaults to `true`

    * `:output_use_bias` - whether to use bias in the output projection.
      Defaults to `true`

    * `:layer_norm` - configuration of the layer norm operation
      at the end of the transformer block. By default the network uses regular
      layer normalization configured with the following options:

        * `:epsilon` - the epsilon used by the layer normalization
          layers. Defaults to `1.0e-5`

      Alternatively a custom 2-arity function may be given. The function
      should add a normalization node to the given Axon node. The function
      also receives layer name prefix as the second argument.

    * `:norm_placement` - controls whether normalization layers should
      be placed before each group of layers (:first) or after each group
      of layers (:last). Defaults to `:last`

    * `:scale_query?` - whether to scale query in the traditional style of
      multi-headed attention. Defaults to `true`

    * `:use_rotary_embedding?` - whether or not to use rotary position embedding
      in multi-headed attention. Defaults to `false`

    * `:position_ids` - input position ids used for rotary embedding

    * `:name` - the prefix for layer names

  ## References

    * [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Figure 1

  """
  def block(hidden_state, opts) do
    validate_required_keys!(opts, [:num_attention_heads, :hidden_size, :ffn])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_attention_heads,
        :hidden_size,
        :ffn,
        attention_mask: Layers.none(),
        attention_head_mask: Layers.none(),
        attention_relative_bias: Layers.none(),
        cross_hidden_state: nil,
        cross_attention_mask: Layers.none(),
        cross_attention_head_mask: Layers.none(),
        block_cache: Layers.none(),
        offset: Layers.none(),
        causal?: false,
        kernel_initializer: :glorot_uniform,
        attention_head_size: nil,
        dropout_rate: 0.0,
        attention_dropout_rate: 0.0,
        query_use_bias: true,
        key_use_bias: true,
        value_use_bias: true,
        output_use_bias: true,
        norm_placement: :last,
        layer_norm: [],
        output_shortcut: true,
        scale_query?: true,
        use_rotary_embedding?: false,
        position_ids: Layers.none()
      ])

    name = opts[:name]
    num_attention_heads = opts[:num_attention_heads]
    hidden_size = opts[:hidden_size]
    ffn = opts[:ffn]
    causal? = opts[:causal?]
    kernel_initializer = opts[:kernel_initializer]
    attention_head_size = opts[:attention_head_size]
    dropout_rate = opts[:dropout_rate]
    attention_dropout_rate = opts[:attention_dropout_rate]
    query_use_bias = opts[:query_use_bias]
    key_use_bias = opts[:key_use_bias]
    value_use_bias = opts[:value_use_bias]
    output_use_bias = opts[:output_use_bias]
    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    attention_relative_bias = opts[:attention_relative_bias]
    cross_hidden_state = opts[:cross_hidden_state]
    cross_attention_mask = opts[:cross_attention_mask]
    cross_attention_head_mask = opts[:cross_attention_head_mask]
    block_cache = opts[:block_cache]
    offset = opts[:offset]
    layer_norm = opts[:layer_norm]
    norm_placement = opts[:norm_placement]
    output_shortcut = opts[:output_shortcut]
    scale_query? = opts[:scale_query?]
    use_rotary_embedding? = opts[:use_rotary_embedding?]
    position_ids = opts[:position_ids]

    ffn_fun =
      case ffn do
        opts when is_list(opts) ->
          validate_required_keys!(opts, [:intermediate_size])
          opts = Keyword.validate!(opts, [:intermediate_size, activation: :gelu])

          &basic_ffn(&1, opts[:intermediate_size], hidden_size,
            activation: opts[:activation],
            kernel_initializer: kernel_initializer,
            dropout_rate: dropout_rate,
            name: &2
          )

        fun when is_function(fun) ->
          fun
      end

    layer_norm_fun =
      case layer_norm do
        opts when is_list(opts) ->
          opts = Keyword.validate!(opts, epsilon: 1.0e-5)

          &Axon.layer_norm(&1, epsilon: opts[:epsilon], name: &2)

        fun when is_function(fun) ->
          fun
      end

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    # Self-attention, shortcut connection, normalization and dropout

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> maybe(norm_placement == :first, fn hidden_state ->
        layer_norm_fun.(hidden_state, join(name, "self_attention_norm"))
      end)

    {hidden_state, attention, self_attention_cache, attention_relative_bias} =
      multi_head_attention(hidden_state, hidden_state, hidden_state,
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        attention_relative_bias: attention_relative_bias,
        attention_cache: self_attention_cache,
        offset: offset,
        causal?: causal?,
        num_heads: num_attention_heads,
        hidden_size: hidden_size,
        kernel_initializer: kernel_initializer,
        attention_head_size: attention_head_size,
        dropout_rate: attention_dropout_rate,
        query_use_bias: query_use_bias,
        key_use_bias: key_use_bias,
        value_use_bias: value_use_bias,
        output_use_bias: output_use_bias,
        scale_query?: scale_query?,
        use_rotary_embedding?: use_rotary_embedding?,
        position_ids: position_ids,
        name: join(name, "self_attention")
      )

    hidden_state =
      hidden_state
      |> Axon.dropout(rate: dropout_rate, name: join(name, "self_attention_dropout"))
      |> Axon.add(shortcut)
      |> maybe(norm_placement == :last, fn hidden_state ->
        layer_norm_fun.(hidden_state, join(name, "self_attention_norm"))
      end)

    # Cross-attention, shortcut connection, normalization and dropout

    {hidden_state, cross_attention, cross_attention_cache} =
      if cross_hidden_state do
        Layers.if_present cross_hidden_state do
          shortcut = hidden_state

          hidden_state =
            hidden_state
            |> maybe(norm_placement == :first, fn hidden_state ->
              layer_norm_fun.(hidden_state, join(name, "cross_attention_norm"))
            end)

          {hidden_state, cross_attention, cross_attention_cache, _cross_attention_relative_bias} =
            multi_head_attention(hidden_state, cross_hidden_state, cross_hidden_state,
              attention_mask: cross_attention_mask,
              attention_head_mask: cross_attention_head_mask,
              attention_cache: cross_attention_cache,
              offset: offset,
              num_heads: num_attention_heads,
              hidden_size: hidden_size,
              kernel_initializer: kernel_initializer,
              attention_head_size: attention_head_size,
              dropout_rate: attention_dropout_rate,
              query_use_bias: query_use_bias,
              key_use_bias: key_use_bias,
              value_use_bias: value_use_bias,
              output_use_bias: output_use_bias,
              scale_query?: scale_query?,
              use_rotary_embedding?: use_rotary_embedding?,
              position_ids: position_ids,
              name: join(name, "cross_attention")
            )

          hidden_state =
            hidden_state
            |> Axon.dropout(rate: dropout_rate, name: join(name, "cross_attention_dropout"))
            |> Axon.add(shortcut)
            |> maybe(norm_placement == :last, fn hidden_state ->
              layer_norm_fun.(hidden_state, join(name, "cross_attention_norm"))
            end)

          {hidden_state, cross_attention, cross_attention_cache}
        else
          {hidden_state, Layers.none(), cross_attention_cache}
        end
      else
        {hidden_state, Layers.none(), cross_attention_cache}
      end

    # Output feed-forward network, shortcut connection, normalization and dropout

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> maybe(norm_placement == :first, fn hidden_state ->
        layer_norm_fun.(hidden_state, join(name, "output_norm"))
      end)
      |> ffn_fun.(join(name, "ffn"))
      |> maybe(output_shortcut, fn hidden_state ->
        Axon.add(hidden_state, shortcut)
      end)
      |> maybe(norm_placement == :last, fn hidden_state ->
        layer_norm_fun.(hidden_state, join(name, "output_norm"))
      end)

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, attention, cross_attention, block_cache, attention_relative_bias}
  end

  defp basic_ffn(x, intermediate_size, output_size, opts) do
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

    * `:causal?` - whether to apply causal attention mask, so that tokens
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

    * `:use_rotary_embedding?` - whether or not to use rotary position embedding
      in multi-headed attention. Defaults to `false`

    * `:position_ids` - input position ids used for rotary embedding

    * `:name` - the prefix for layer names

  ## References

    * [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Figure 2

  """
  def multi_head_attention(query, key, value, opts) do
    validate_required_keys!(opts, [:num_heads, :hidden_size])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_heads,
        :hidden_size,
        attention_mask: Layers.none(),
        attention_head_mask: Layers.none(),
        attention_relative_bias: Layers.none(),
        attention_cache: Layers.none(),
        offset: Layers.none(),
        causal?: false,
        scale_query?: true,
        kernel_initializer: :glorot_uniform,
        dropout_rate: 0.0,
        attention_head_size: nil,
        query_use_bias: true,
        key_use_bias: true,
        value_use_bias: true,
        output_use_bias: true,
        use_rotary_embedding?: false,
        position_ids: Layers.none()
      ])

    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    attention_cache = opts[:attention_cache]
    offset = opts[:offset]

    name = opts[:name]
    num_heads = opts[:num_heads]
    hidden_size = opts[:hidden_size]
    kernel_initializer = opts[:kernel_initializer]
    causal? = opts[:causal?]
    scale_query? = opts[:scale_query?]
    dropout_rate = opts[:dropout_rate]
    use_rotary_embedding? = opts[:use_rotary_embedding?]
    position_ids = opts[:position_ids]

    query_use_bias = opts[:query_use_bias]
    key_use_bias = opts[:key_use_bias]
    value_use_bias = opts[:value_use_bias]
    output_use_bias = opts[:output_use_bias]

    attention_relative_bias = opts[:attention_relative_bias]

    inner_size =
      if attention_head_size = opts[:attention_head_size] do
        num_heads * attention_head_size
      else
        hidden_size
      end

    query =
      query
      |> Axon.dense(inner_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "query"),
        use_bias: query_use_bias
      )
      |> Layers.split_heads(num_heads)

    key =
      key
      |> Axon.dense(inner_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "key"),
        use_bias: key_use_bias
      )
      |> Layers.split_heads(num_heads)

    value =
      value
      |> Axon.dense(inner_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "value"),
        use_bias: value_use_bias
      )
      |> Layers.split_heads(num_heads)

    {query, key} =
      if use_rotary_embedding? do
        Layers.rotary_embedding(
          query,
          key,
          value,
          position_ids,
          div(hidden_size, num_heads),
          name: join(name, "rotary_embedding")
        )
      else
        {query, key}
      end

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, attention_cache, offset)

    attention_mask = Layers.expand_attention_mask(attention_mask)

    attention_mask =
      if causal? do
        Layers.Decoder.apply_causal_mask(attention_mask, query, offset)
      else
        attention_mask
      end

    attention_bias = Layers.attention_bias(attention_mask)

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

    attention_bias =
      Layers.if_present attention_relative_bias do
        Axon.add(attention_bias, attention_relative_bias)
      else
        attention_bias
      end

    attention_weights =
      Layers.attention_weights(query, key, attention_bias, scale_query?: scale_query?)
      |> Axon.dropout(rate: dropout_rate)
      |> Layers.apply_attention_head_mask(attention_head_mask)

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()
      |> Axon.dense(hidden_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "output"),
        use_bias: output_use_bias
      )

    {attention_output, attention_weights, attention_cache, attention_relative_bias}
  end

  defp maybe(term, false, _fun), do: term
  defp maybe(term, true, fun), do: fun.(term)

  defp validate_required_keys!(opts, keys) do
    case keys -- Keyword.keys(opts) do
      [] -> :ok
      missing -> raise ArgumentError, "missing required options: #{inspect(missing)}"
    end
  end
end
