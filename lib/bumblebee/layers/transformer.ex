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

    * `:rotary_embedding` - configuration of rotary embedding. Can be:
      - a keyword list (applied to all blocks)
      - a function that takes the block index and returns the configuration

    * `:attention_window_size` - sliding window attention configuration. Can be:
      - `nil` for global attention (default)
      - a `{left, right}` tuple (applied to all blocks)
      - a function that takes the block index and returns `nil` or `{left, right}`.
        This enables per-layer attention patterns like Gemma 3's alternating
        local/global attention (5 local layers followed by 1 global layer)

    * `:name` - the prefix for layer names

  For all other options (including required options) see `block/2`.
  Note that `:attention_head_mask`, `:cross_attention_head_mask` should
  have an additional leading axis corresponding to the number of blocks,
  since each block gets its own mask. Similarly `:cache` is a tuple of
  block caches.
  """
  def blocks(hidden_state, opts) do
    validate_required_keys!(opts, [:num_blocks, :num_attention_heads, :hidden_size, :ffn])

    # Note: :attention_window_size is NOT in block_opts_keys because it's handled
    # specially (supports per-layer function) and passed explicitly to block/2
    block_opts_keys = [
      :num_attention_heads,
      :num_key_value_heads,
      :causal,
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
      :block_type,
      :attention_scale,
      :query_norm,
      :key_norm
    ]

    opts =
      Keyword.validate!(
        opts,
        block_opts_keys ++
          [
            :name,
            :num_blocks,
            :rotary_embedding,
            :attention_window_size,
            attention_mask: Layers.none(),
            attention_head_mask: Layers.none(),
            attention_relative_bias: nil,
            share_attention_relative_bias: false,
            cross_hidden_state: nil,
            cross_attention_mask: Layers.none(),
            cross_attention_head_mask: Layers.none(),
            cache: Layers.none()
          ]
      )

    name = opts[:name]
    num_blocks = opts[:num_blocks]

    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    cross_hidden_state = opts[:cross_hidden_state]
    cross_attention_mask = opts[:cross_attention_mask]
    cross_attention_head_mask = opts[:cross_attention_head_mask]
    cache = opts[:cache]
    rotary_embedding = opts[:rotary_embedding]
    attention_window_size = opts[:attention_window_size]

    block_opts = Keyword.take(opts, block_opts_keys)

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cross_attentions: Axon.container({}),
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

          block_rotary_embedding =
            case rotary_embedding do
              nil -> nil
              fun when is_function(fun, 1) -> fun.(idx)
              config when is_list(config) -> config
            end

          # Support per-layer attention window size for models like Gemma 3
          # that alternate between local (sliding window) and global attention
          block_attention_window_size =
            case attention_window_size do
              nil -> nil
              fun when is_function(fun, 1) -> fun.(idx)
              size -> size
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
                rotary_embedding: block_rotary_embedding,
                attention_window_size: block_attention_window_size,
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

    * `:causal` - whether the self-attention block should be causal.
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
      also receives the layer name prefix as the second argument.

    * `:block_type` - controls which configuration of the block to use,
      one of:

        * `:standard` (default) - the original transformer block

        * `:norm_first` - same as `:standard`, but with normalization layers
          placed before each group of layers, rather than after

        * `:parallel` - block with attention and FFN independently (in parallel).
          This type doesn't support cross-attention

      Alternatively a custom 3-arity function may be given. The function
      receives the input hidden state, a map with block steps and a
      name to prefix any additional layers.

    * `:attention_window_size` - when set, enables sliding window attention.
      Should be a `{left, right}` tuple with window size on each side

    * `:attention_scale` - the scaling factor applied to the attention weights.
      Defaults to $\frac{1}{\sqrt{d}}$.

    * `:rotary_embedding` - configuration of rotary embedding. If set,
      will apply rotary position embedding with the given options. Valid
      options are:

        * `:position_ids` (required) - input position ids used for the
          embedding

        * `:max_positions` - the maximum number of distinct positions

        * `:base` - base for computing rotary embedding frequency. Defaults
        to `10_000`.

        * `:percentage` - percentage of hidden dimensions to allocate to rotary embeddings.
        Defaults to `1.0`.

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
        :num_key_value_heads,
        attention_mask: Layers.none(),
        attention_head_mask: Layers.none(),
        attention_relative_bias: Layers.none(),
        cross_hidden_state: nil,
        cross_attention_mask: Layers.none(),
        cross_attention_head_mask: Layers.none(),
        block_cache: Layers.none(),
        offset: Layers.none(),
        causal: false,
        kernel_initializer: :glorot_uniform,
        attention_head_size: nil,
        dropout_rate: 0.0,
        attention_dropout_rate: 0.0,
        query_use_bias: true,
        key_use_bias: true,
        value_use_bias: true,
        output_use_bias: true,
        block_type: :standard,
        layer_norm: [],
        attention_window_size: nil,
        attention_scale: nil,
        rotary_embedding: nil,
        query_norm: nil,
        key_norm: nil
      ])

    name = opts[:name]
    num_attention_heads = opts[:num_attention_heads]
    num_key_value_heads = opts[:num_key_value_heads] || num_attention_heads
    hidden_size = opts[:hidden_size]
    ffn = opts[:ffn]
    causal = opts[:causal]
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
    block_type = opts[:block_type]
    attention_window_size = opts[:attention_window_size]
    attention_scale = opts[:attention_scale]
    rotary_embedding = opts[:rotary_embedding]
    query_norm = opts[:query_norm]
    key_norm = opts[:key_norm]

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

    self_attention_norm = &layer_norm_fun.(&1, join(name, "self_attention_norm"))

    self_attention = fn hidden_state ->
      {hidden_state, attention, self_attention_cache, attention_relative_bias} =
        multi_head_attention(hidden_state, hidden_state, hidden_state,
          attention_mask: attention_mask,
          attention_head_mask: attention_head_mask,
          attention_relative_bias: attention_relative_bias,
          attention_cache: self_attention_cache,
          offset: offset,
          causal: causal,
          num_heads: num_attention_heads,
          num_key_value_heads: num_key_value_heads,
          hidden_size: hidden_size,
          kernel_initializer: kernel_initializer,
          attention_head_size: attention_head_size,
          dropout_rate: attention_dropout_rate,
          query_use_bias: query_use_bias,
          key_use_bias: key_use_bias,
          value_use_bias: value_use_bias,
          output_use_bias: output_use_bias,
          attention_window_size: attention_window_size,
          attention_scale: attention_scale,
          rotary_embedding: rotary_embedding,
          query_norm: query_norm,
          key_norm: key_norm,
          name: join(name, "self_attention")
        )

      hidden_state =
        Axon.dropout(hidden_state, rate: dropout_rate, name: join(name, "self_attention_dropout"))

      {hidden_state, {attention, self_attention_cache, attention_relative_bias}}
    end

    # Cross-attention, shortcut connection, normalization and dropout

    cross_attention_maybe = fn hidden_state, fun ->
      if cross_hidden_state do
        Layers.if_present cross_hidden_state do
          fun.(hidden_state)
        else
          {hidden_state, {Layers.none(), cross_attention_cache}}
        end
      else
        {hidden_state, {Layers.none(), cross_attention_cache}}
      end
    end

    cross_attention_norm = &layer_norm_fun.(&1, join(name, "cross_attention_norm"))

    cross_attention = fn hidden_state ->
      {hidden_state, cross_attention, cross_attention_cache, _cross_attention_relative_bias} =
        multi_head_attention(hidden_state, cross_hidden_state, cross_hidden_state,
          attention_mask: cross_attention_mask,
          attention_head_mask: cross_attention_head_mask,
          attention_cache: cross_attention_cache,
          offset: offset,
          num_heads: num_attention_heads,
          num_key_value_heads: num_key_value_heads,
          hidden_size: hidden_size,
          kernel_initializer: kernel_initializer,
          attention_head_size: attention_head_size,
          dropout_rate: attention_dropout_rate,
          query_use_bias: query_use_bias,
          key_use_bias: key_use_bias,
          value_use_bias: value_use_bias,
          output_use_bias: output_use_bias,
          attention_window_size: attention_window_size,
          attention_scale: attention_scale,
          rotary_embedding: rotary_embedding,
          name: join(name, "cross_attention")
        )

      hidden_state =
        Axon.dropout(
          hidden_state,
          rate: dropout_rate,
          name: join(name, "cross_attention_dropout")
        )

      {hidden_state, {cross_attention, cross_attention_cache}}
    end

    # Output feed-forward network, shortcut connection, normalization and dropout

    output_norm = &layer_norm_fun.(&1, join(name, "output_norm"))

    ffn = &ffn_fun.(&1, join(name, "ffn"))

    block_impl =
      case block_type do
        type when is_atom(type) -> &block_impl(type, &1, &2, &3)
        fun when is_function(fun) -> fun
      end

    {hidden_state, attention_info, cross_attention_info} =
      block_impl.(
        hidden_state,
        %{
          self_attention_norm: self_attention_norm,
          self_attention: self_attention,
          cross_attention_maybe: cross_attention_maybe,
          cross_attention_norm: cross_attention_norm,
          cross_attention: cross_attention,
          output_norm: output_norm,
          ffn: ffn
        },
        name
      )

    {attention, self_attention_cache, attention_relative_bias} = attention_info
    {cross_attention, cross_attention_cache} = cross_attention_info

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, attention, cross_attention, block_cache, attention_relative_bias}
  end

  defp block_impl(:standard, hidden_state, steps, _name) do
    shortcut = hidden_state

    {hidden_state, attention_info} = steps.self_attention.(hidden_state)

    hidden_state =
      hidden_state
      |> Axon.add(shortcut)
      |> steps.self_attention_norm.()

    {hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn hidden_state ->
        shortcut = hidden_state

        {hidden_state, cross_attention_info} = steps.cross_attention.(hidden_state)

        hidden_state =
          hidden_state
          |> Axon.add(shortcut)
          |> steps.cross_attention_norm.()

        {hidden_state, cross_attention_info}
      end)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> steps.ffn.()
      |> Axon.add(shortcut)
      |> steps.output_norm.()

    {hidden_state, attention_info, cross_attention_info}
  end

  defp block_impl(:norm_first, hidden_state, steps, _name) do
    shortcut = hidden_state

    {hidden_state, attention_info} =
      hidden_state
      |> steps.self_attention_norm.()
      |> steps.self_attention.()

    hidden_state = Axon.add(hidden_state, shortcut)

    {hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn hidden_state ->
        shortcut = hidden_state

        {hidden_state, cross_attention_info} =
          hidden_state
          |> steps.cross_attention_norm.()
          |> steps.cross_attention.()

        hidden_state = Axon.add(hidden_state, shortcut)

        {hidden_state, cross_attention_info}
      end)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> steps.output_norm.()
      |> steps.ffn.()
      |> Axon.add(shortcut)

    {hidden_state, attention_info, cross_attention_info}
  end

  defp block_impl(:parallel, hidden_state, steps, _name) do
    shortcut = hidden_state

    {attention_hidden_state, attention_info} =
      hidden_state
      |> steps.self_attention_norm.()
      |> steps.self_attention.()

    {_hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn _hidden_state ->
        raise "cross attention not supported"
      end)

    ffn_hidden_state =
      hidden_state
      |> steps.output_norm.()
      |> steps.ffn.()

    hidden_state = Axon.add([shortcut, attention_hidden_state, ffn_hidden_state])

    {hidden_state, attention_info, cross_attention_info}
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
    inner_size = num_heads * attention_head_size
    inner_kv_size = num_key_value_heads * attention_head_size

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
      |> Axon.dense(inner_kv_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "key"),
        use_bias: key_use_bias
      )
      |> Layers.split_heads(num_key_value_heads)

    value =
      value
      |> Axon.dense(inner_kv_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "value"),
        use_bias: value_use_bias
      )
      |> Layers.split_heads(num_key_value_heads)

    # Apply query and key normalization if configured (before rotary embedding)
    query =
      if query_norm do
        query_norm.(query, join(name, "query_norm"))
      else
        query
      end

    key =
      if key_norm do
        key_norm.(key, join(name, "key_norm"))
      else
        key
      end

    {query, key} =
      case rotary_embedding do
        opts when is_list(opts) ->
          validate_required_keys!(opts, [:position_ids])

          opts =
            Keyword.validate!(opts, [
              :position_ids,
              :max_positions,
              :scaling_strategy,
              base: 10_000,
              percentage: 1.0
            ])

          {position_ids, opts} = Keyword.pop(opts, :position_ids)
          {percentage, opts} = Keyword.pop(opts, :percentage)

          size = trunc(attention_head_size * percentage)

          rotary_opts = [name: join(name, "rotary_embedding")] ++ opts

          if size == attention_head_size do
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

        nil ->
          {query, key}
      end

    num_key_value_groups = div(num_heads, num_key_value_heads)
    key = repeat_states(key, num_key_value_groups)
    value = repeat_states(value, num_key_value_groups)

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
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(hidden_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "output"),
        use_bias: output_use_bias
      )

    {attention_output, attention_weights, attention_cache, attention_relative_bias}
  end

  defp repeat_states(state, 1), do: state

  defp repeat_states(state, times) do
    Layers.repeat_interleave(state, times, axis: 2)
  end

  defp validate_required_keys!(opts, keys) do
    case keys -- Keyword.keys(opts) do
      [] -> :ok
      missing -> raise ArgumentError, "missing required options: #{inspect(missing)}"
    end
  end
end
