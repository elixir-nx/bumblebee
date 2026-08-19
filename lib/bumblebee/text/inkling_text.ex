defmodule Bumblebee.Text.InklingText do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 201_024,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      unpadded_vocab_size: [
        default: nil,
        doc: """
        the number of rows that the language modeling head actually holds, when it is not
        padded to `:vocab_size`. Logits beyond this number are dropped
        """
      ],
      max_positions: [
        default: 131_072,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process
        """
      ],
      hidden_size: [
        default: 6144,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 24_576,
        doc: "the dimensionality of intermediate layers in the dense feed-forward blocks"
      ],
      moe_intermediate_size: [
        default: 3072,
        doc: "the dimensionality of intermediate layers in each mixture-of-experts expert"
      ],
      num_blocks: [
        default: 66,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 64,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 8,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      attention_head_size: [
        default: 128,
        doc: "the size of the key, value, and query projection per attention head"
      ],
      block_types: [
        default: nil,
        doc: """
        a list with the attention type of each block, either `:hybrid` for blocks using full
        attention, or `:hybrid_sliding` for blocks using sliding window attention. When `nil`,
        every block whose index is not a multiple of 6 (counting from one) uses sliding window
        attention
        """
      ],
      attention_window_size: [
        default: 512,
        doc: "the size of the attention window for blocks using sliding window attention"
      ],
      relative_size: [
        default: 16,
        doc: """
        the per-head dimensionality of the relative states, which are mixed into the relative
        position bias. Inkling has no rotary embedding and positions enter the model only
        through this bias
        """
      ],
      relative_extent: [
        default: 1024,
        doc: """
        the backward distance, in tokens, over which the relative position bias is applied.
        The bias is zero beyond it
        """
      ],
      log_scaling_position: [
        default: nil,
        doc: """
        the position from which the attention logits start being scaled up logarithmically in
        the blocks using full attention. When `nil`, the scaling is disabled
        """
      ],
      log_scaling_alpha: [
        default: 0.1,
        doc: "the strength of the logarithmic attention logit scaling"
      ],
      conv_kernel_size: [
        default: 4,
        doc: "the kernel size of the causal short convolutions"
      ],
      mlp_block_types: [
        default: nil,
        doc: """
        a list with the feed-forward network type of each block, either `:dense` or `:sparse`.
        When `nil`, every block uses a mixture-of-experts block
        """
      ],
      num_experts: [
        default: 256,
        doc: "the number of routed experts in each mixture-of-experts block"
      ],
      num_experts_per_token: [
        default: 6,
        doc: "the number of experts that each token is routed to"
      ],
      num_shared_experts: [
        default: 2,
        doc: "the number of shared experts, which are applied to all tokens"
      ],
      route_scale: [
        default: 8.0,
        doc: "the constant that the routing weights are multiplied by"
      ],
      logits_scale: [
        default: 24.0,
        doc: """
        the muP width multiplier that the final hidden states are divided by before the
        language modeling head
        """
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      params_prefix: [
        default: nil,
        doc: nil
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: nil)

  @moduledoc """
  The text model of the Inkling model family.

  Inkling is a multimodal mixture-of-experts model and this module
  implements its text decoder. The decoder has several distinctive
  features:

    * there is no rotary embedding, positional information enters the
      model through a learnt relative position bias, conditioned on the
      hidden state

    * causal short convolutions are applied to the key and value
      projections, as well as to the output of both the attention and the
      feed-forward network

    * the mixture-of-experts router scores the shared experts alongside
      the routed ones, so that they act as a sink in the softmax over
      expert weights

  ## Architectures

    * `:base` - plain Inkling text model without any head on top

    * `:for_causal_language_modeling` - Inkling text model with a language
      modeling head. The head returns logits for each token in the
      original sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Nx.Defn
  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers
  alias Bumblebee.Layers.Moe

  @impl true
  def architectures(), do: [:base, :for_causal_language_modeling]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    key_value_size = spec.num_key_value_heads * spec.attention_head_size
    window_size = spec.conv_kernel_size - 1

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks,
      extra_window_states: [
        key_convolution: {window_size, key_value_size},
        value_convolution: {window_size, key_value_size},
        attention_convolution: {window_size, spec.hidden_size},
        ffn_convolution: {window_size, spec.hidden_size}
      ]
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings = embedder(inputs["input_ids"], inputs["input_embeddings"], spec, name: "embedder")

    decoder_outputs =
      decoder(embeddings, inputs["attention_mask"], inputs["cache"], spec, name: "decoder")

    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    # The normalization is a part of the embedding lookup, so it does
    # not apply to embeddings passed by the caller, such as the image
    # embeddings that a multimodal model splices in
    Layers.default input_embeddings do
      input_ids
      |> Axon.embedding(spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
      |> Layers.rms_norm(
        name: join(name, "norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )
    end
  end

  # Inkling blocks apply a short convolution to the output of both the
  # attention and the feed-forward network, and those convolutions have
  # their own cache entries, so we build the blocks explicitly, rather
  # than using `Bumblebee.Layers.Transformer.blocks/2`
  defp decoder(hidden_state, attention_mask, cache, spec, opts) do
    name = opts[:name]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    block_types = block_types(spec)
    mlp_block_types = mlp_block_types(spec)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cache: cache
    }

    outputs =
      for idx <- 0..(spec.num_blocks - 1), reduce: state do
        state ->
          block_name = join(join(name, "blocks"), idx)

          block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

          {attention_cache, cross_attention_cache} =
            Layers.Decoder.get_attention_caches(block_cache)

          shortcut = state.hidden_state

          hidden_state =
            Layers.rms_norm(state.hidden_state,
              name: join(block_name, "self_attention_norm"),
              epsilon: spec.layer_norm_epsilon,
              upcast: :all
            )

          {hidden_state, attention, attention_cache} =
            attention(
              hidden_state,
              attention_mask,
              attention_cache,
              offset,
              Enum.at(block_types, idx),
              spec,
              name: join(block_name, "self_attention")
            )

          {hidden_state, attention_cache} =
            short_convolution(
              hidden_state,
              attention_mask,
              attention_cache,
              offset,
              :attention_convolution,
              spec.hidden_size,
              spec,
              name: join(block_name, "self_attention_convolution")
            )

          hidden_state = Axon.add(hidden_state, shortcut)

          shortcut = hidden_state

          hidden_state =
            hidden_state
            |> Layers.rms_norm(
              name: join(block_name, "output_norm"),
              epsilon: spec.layer_norm_epsilon,
              upcast: :all
            )
            |> ffn(Enum.at(mlp_block_types, idx), spec, name: join(block_name, "ffn"))

          {hidden_state, attention_cache} =
            short_convolution(
              hidden_state,
              attention_mask,
              attention_cache,
              offset,
              :ffn_convolution,
              spec.hidden_size,
              spec,
              name: join(block_name, "ffn_convolution")
            )

          hidden_state = Axon.add(hidden_state, shortcut)

          block_cache =
            Layers.Decoder.put_attention_caches(
              block_cache,
              attention_cache,
              cross_attention_cache
            )

          %{
            hidden_state: hidden_state,
            hidden_states: Layers.append(state.hidden_states, hidden_state),
            attentions: Layers.append(state.attentions, attention),
            cache: Layers.Decoder.put_block_cache(state.cache, idx, block_cache)
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  defp attention(hidden_state, attention_mask, attention_cache, offset, block_type, spec, opts) do
    name = opts[:name]

    sliding? = block_type == :hybrid_sliding

    num_heads = spec.num_attention_heads
    num_key_value_heads = spec.num_key_value_heads
    head_size = spec.attention_head_size

    query =
      hidden_state
      |> Axon.dense(num_heads * head_size, name: join(name, "query"), use_bias: false)
      |> Layers.split_heads(num_heads)
      |> Layers.rms_norm(
        name: join(name, "query_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    key =
      Axon.dense(hidden_state, num_key_value_heads * head_size,
        name: join(name, "key"),
        use_bias: false
      )

    value =
      Axon.dense(hidden_state, num_key_value_heads * head_size,
        name: join(name, "value"),
        use_bias: false
      )

    key_value_size = num_key_value_heads * head_size

    {key, attention_cache} =
      short_convolution(
        key,
        attention_mask,
        attention_cache,
        offset,
        :key_convolution,
        key_value_size,
        spec,
        name: join(name, "key_convolution")
      )

    {value, attention_cache} =
      short_convolution(
        value,
        attention_mask,
        attention_cache,
        offset,
        :value_convolution,
        key_value_size,
        spec,
        name: join(name, "value_convolution")
      )

    key =
      key
      |> Layers.split_heads(num_key_value_heads)
      |> Layers.rms_norm(
        name: join(name, "key_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    value = Layers.split_heads(value, num_key_value_heads)

    relative_states =
      hidden_state
      |> Axon.dense(num_heads * spec.relative_size,
        name: join(name, "relative"),
        use_bias: false
      )
      |> Layers.split_heads(num_heads)

    num_key_value_groups = div(num_heads, num_key_value_heads)
    key = repeat_states(key, num_key_value_groups)
    value = repeat_states(value, num_key_value_groups)

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, attention_cache, offset)

    relative_extent = if sliding?, do: spec.attention_window_size, else: spec.relative_extent

    relative_bias =
      relative_bias(relative_states, key, offset, spec,
        relative_extent: relative_extent,
        name: join(name, "relative_bias")
      )

    {query, relative_bias} =
      if not sliding? and spec.log_scaling_position do
        log_scaling(query, relative_bias, offset,
          position: spec.log_scaling_position,
          alpha: spec.log_scaling_alpha
        )
      else
        {query, relative_bias}
      end

    window_size =
      if sliding? do
        # The window includes the current position, so the maximum
        # distance to an attended position is one less
        {spec.attention_window_size - 1, 0}
      end

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        attention_mask,
        Layers.none(),
        relative_bias,
        offset,
        causal: true,
        window_size: window_size,
        # Query and key are normalized per head, hence 1/d rather than
        # the usual 1/sqrt(d)
        scale: 1.0 / head_size
      )

    attention_output =
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size, name: join(name, "output"), use_bias: false)

    {attention_output, attention_weights, attention_cache}
  end

  defp repeat_states(state, 1), do: state
  defp repeat_states(state, times), do: Layers.repeat_interleave(state, times, axis: 2)

  # A learnt bank of bias-vs-distance profiles. Each token mixes them
  # into a single bias value per backward distance
  defp relative_bias(relative_states, key, offset, spec, opts) do
    name = opts[:name]
    relative_extent = opts[:relative_extent]

    projection =
      Axon.param("kernel", fn _ -> {spec.relative_size, relative_extent} end,
        initializer: kernel_initializer(spec)
      )

    logits =
      Axon.layer(&relative_logits_impl/3, [relative_states, projection],
        name: name,
        op_name: :relative_position_bias
      )

    Axon.layer(&relative_bias_impl/4, [logits, key, Axon.optional(offset)],
      relative_extent: relative_extent
    )
  end

  defnp relative_logits_impl(relative_states, projection, _opts \\ []) do
    # {batch_size, num_heads, sequence_length, relative_extent}
    relative_states
    |> Nx.dot([3], projection, [0])
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defnp relative_bias_impl(logits, key, offset, opts \\ []) do
    opts = keyword!(opts, [:relative_extent, mode: :inference])

    relative_extent = opts[:relative_extent]

    query_length = Nx.axis_size(logits, 2)
    key_length = Nx.axis_size(key, 1)

    distance = Nx.iota({query_length, 1}) + ensure_offset(offset) - Nx.iota({1, key_length})

    index =
      distance
      |> Nx.clip(0, relative_extent - 1)
      |> Nx.reshape({1, 1, query_length, key_length})
      |> Nx.broadcast(
        {Nx.axis_size(logits, 0), Nx.axis_size(logits, 1), query_length, key_length}
      )

    bias = Nx.take_along_axis(logits, index, axis: -1)

    within_extent =
      Nx.logical_and(Nx.greater_equal(distance, 0), Nx.less(distance, relative_extent))

    within_extent =
      within_extent
      |> Nx.reshape({1, 1, query_length, key_length})
      |> Nx.broadcast(Nx.shape(bias))

    Nx.select(within_extent, bias, 0.0)
  end

  # In blocks using full attention, the attention logits are scaled up
  # logarithmically past a certain position, so that the attention stays
  # sharp for very long sequences
  defp log_scaling(query, relative_bias, offset, opts) do
    scale =
      Axon.layer(&log_scaling_impl/3, [query, Axon.optional(offset)],
        op_name: :log_scaling,
        position: opts[:position],
        alpha: opts[:alpha]
      )

    query =
      Axon.layer(
        fn query, scale, _opts -> Nx.multiply(query, Nx.reshape(scale, {1, :auto, 1, 1})) end,
        [query, scale]
      )

    relative_bias =
      Axon.layer(
        fn bias, scale, _opts -> Nx.multiply(bias, Nx.reshape(scale, {1, 1, :auto, 1})) end,
        [relative_bias, scale]
      )

    {query, relative_bias}
  end

  defnp log_scaling_impl(query, offset, opts \\ []) do
    opts = keyword!(opts, [:position, :alpha, mode: :inference])

    offset =
      case offset do
        %Axon.None{} -> 0
        offset -> offset
      end

    positions = Nx.iota({Nx.axis_size(query, 1)}) + offset + 1

    1.0 + opts[:alpha] * Nx.log(Nx.max(positions / opts[:position], 1.0))
  end

  # A depthwise causal convolution over the sequence axis, with a
  # shortcut connection
  defp short_convolution(
         hidden_state,
         attention_mask,
         attention_cache,
         offset,
         cache_name,
         channels,
         spec,
         opts
       ) do
    {full_hidden_state, attention_cache} =
      Layers.Decoder.cached_window_state(hidden_state, cache_name, attention_cache)

    output =
      hidden_state
      |> Layers.causal_depthwise_conv1d(full_hidden_state, attention_mask, offset,
        channels: channels,
        kernel_size: spec.conv_kernel_size,
        name: opts[:name]
      )
      |> Axon.add(hidden_state)

    {output, attention_cache}
  end

  defnp ensure_offset(offset) do
    case offset do
      %Axon.None{} -> 0
      offset -> offset
    end
  end

  defp ffn(hidden_state, :dense, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Moe.gated_ffn(spec.intermediate_size, spec.hidden_size,
      activation: spec.activation,
      kernel_initializer: kernel_initializer(spec),
      name: name
    )
    |> global_scale(join(name, "scale"))
  end

  defp ffn(hidden_state, :sparse, spec, opts) do
    name = opts[:name]

    {weights, shared_weights} =
      shared_sink_router(hidden_state, spec, name: join(name, "router"))

    routed =
      Moe.experts(hidden_state, weights,
        num_experts: spec.num_experts,
        hidden_size: spec.hidden_size,
        intermediate_size: spec.moe_intermediate_size,
        activation: spec.activation,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "experts")
      )

    shared =
      Moe.experts(hidden_state, shared_weights,
        num_experts: spec.num_shared_experts,
        hidden_size: spec.hidden_size,
        intermediate_size: spec.moe_intermediate_size,
        activation: spec.activation,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "shared_experts")
      )

    Axon.add(routed, shared)
  end

  # The router scores the shared experts alongside the routed ones, so
  # that they act as a sink in the softmax over the expert weights
  defp shared_sink_router(hidden_state, spec, opts) do
    name = opts[:name]

    num_total_experts = spec.num_experts + spec.num_shared_experts

    kernel =
      Axon.param(
        "kernel",
        fn shape -> {elem(shape, tuple_size(shape) - 1), num_total_experts} end,
        initializer: kernel_initializer(spec)
      )

    bias =
      Axon.param("score_correction_bias", fn _ -> {spec.num_experts} end, initializer: :zeros)

    scale = Axon.param("scale", fn _ -> {1} end, initializer: :ones)

    Axon.layer(&shared_sink_router_impl/5, [hidden_state, kernel, bias, scale],
      name: name,
      op_name: :moe_shared_sink_router,
      num_experts: spec.num_experts,
      num_experts_per_token: spec.num_experts_per_token,
      num_shared_experts: spec.num_shared_experts,
      route_scale: spec.route_scale
    )
    |> Layers.unwrap_tuple(2)
  end

  defnp shared_sink_router_impl(hidden_state, kernel, score_correction_bias, scale, opts \\ []) do
    opts =
      keyword!(opts, [
        :num_experts,
        :num_experts_per_token,
        :num_shared_experts,
        :route_scale,
        mode: :inference
      ])

    num_experts = opts[:num_experts]
    num_shared_experts = opts[:num_shared_experts]

    hidden_state = Nx.as_type(hidden_state, :f32)
    kernel = Nx.as_type(kernel, :f32)

    logits = Nx.dot(hidden_state, [-1], kernel, [0])

    routed_logits = Nx.slice_along_axis(logits, 0, num_experts, axis: -1)
    shared_logits = Nx.slice_along_axis(logits, num_experts, num_shared_experts, axis: -1)

    selection_scores = Nx.sigmoid(routed_logits) + Nx.as_type(score_correction_bias, :f32)

    order = Nx.argsort(selection_scores, axis: -1, direction: :desc, stable: true)
    rank = Nx.argsort(order, axis: -1, stable: true)
    selected = Nx.less(rank, opts[:num_experts_per_token])

    routed_log_probabilities = Axon.Activations.log_sigmoid(routed_logits)
    shared_log_probabilities = Axon.Activations.log_sigmoid(shared_logits)

    # The normalization is over the selected experts and all of the
    # shared ones
    selected_log_probabilities =
      Nx.select(selected, routed_log_probabilities, Nx.Constants.neg_infinity(:f32))

    denominator =
      Nx.concatenate([selected_log_probabilities, shared_log_probabilities], axis: -1)
      |> Nx.logsumexp(axes: [-1], keep_axes: true)

    scale = Nx.as_type(scale, :f32) * opts[:route_scale]

    weights =
      Nx.exp(routed_log_probabilities - denominator) * Nx.as_type(selected, :f32) * scale

    shared_weights = Nx.exp(shared_log_probabilities - denominator) * scale

    {weights, shared_weights}
  end

  defp global_scale(hidden_state, name) do
    scale = Axon.param("scale", fn _ -> {1} end, initializer: :ones)

    Axon.layer(
      fn hidden_state, scale, _opts -> Nx.multiply(hidden_state, scale) end,
      [hidden_state, scale],
      name: name,
      op_name: :global_scale
    )
  end

  defp block_types(spec) do
    spec.block_types ||
      for idx <- 0..(spec.num_blocks - 1) do
        if rem(idx + 1, 6) == 0, do: :hybrid, else: :hybrid_sliding
      end
  end

  defp mlp_block_types(spec) do
    spec.mlp_block_types || List.duplicate(:sparse, spec.num_blocks)
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.nx(&Nx.divide(&1, spec.logits_scale))
    |> Layers.dense_transposed(spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> then(fn logits ->
      case spec.unpadded_vocab_size do
        nil -> logits
        size when size >= spec.vocab_size -> logits
        size -> Axon.nx(logits, &Nx.slice_along_axis(&1, 0, size, axis: -1))
      end
    end)
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      params_prefix =
        if data["text_config"], do: [params_prefix: "model.language_model"], else: []

      data = Shared.text_config(data)

      block_type_converter =
        list(mapping(%{"hybrid" => :hybrid, "hybrid_sliding" => :hybrid_sliding}))

      mlp_block_type_converter = list(mapping(%{"dense" => :dense, "sparse" => :sparse}))

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          unpadded_vocab_size: {"unpadded_vocab_size", optional(number())},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", number()},
          moe_intermediate_size: {"moe_intermediate_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          block_types: {"layer_types", optional(block_type_converter)},
          attention_window_size: {"sliding_window_size", number()},
          relative_size: {"d_rel", number()},
          relative_extent: {"rel_extent", number()},
          log_scaling_position: {"log_scaling_n_floor", optional(number())},
          log_scaling_alpha: {"log_scaling_alpha", number()},
          conv_kernel_size: {"conv_kernel_size", number()},
          mlp_block_types: {"mlp_layer_types", optional(mlp_block_type_converter)},
          num_experts: {"n_routed_experts", number()},
          num_experts_per_token: {"num_experts_per_tok", number()},
          num_shared_experts: {"n_shared_experts", number()},
          route_scale: {"route_scale", number()},
          logits_scale: {"logits_mup_width_multiplier", number()},
          activation: {"hidden_act", activation()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec) ++ params_prefix

      opts =
        case data["sconv_kernel_size"] do
          nil -> opts
          size -> Keyword.put(opts, :conv_kernel_size, size)
        end

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.Layers.Moe

    import Bumblebee.Utils.Model, only: [join: 2]

    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "embedder.norm" => "model.embed_norm",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.relative" => "model.layers.{n}.self_attn.r_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention.query_norm" => "model.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" => "model.layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "output_norm" => "model.norm",
        "language_modeling_head.output" => "lm_head"
      }
      |> Map.merge(convolution_params_mapping())
      |> Map.merge(relative_bias_params_mapping())
      |> Map.merge(scale_params_mapping())
      |> Map.merge(expert_params_mapping(spec))
      |> Bumblebee.Shared.replace_params_mapping_source_prefix(spec.params_prefix)
    end

    defp convolution_params_mapping() do
      for {target, source} <- [
            {"decoder.blocks.{n}.self_attention.key_convolution",
             "model.layers.{n}.self_attn.k_sconv.conv1d"},
            {"decoder.blocks.{n}.self_attention.value_convolution",
             "model.layers.{n}.self_attn.v_sconv.conv1d"},
            {"decoder.blocks.{n}.self_attention_convolution",
             "model.layers.{n}.attn_sconv.conv1d"},
            {"decoder.blocks.{n}.ffn_convolution", "model.layers.{n}.mlp_sconv.conv1d"}
          ],
          into: %{} do
        {target,
         %{
           "kernel" => {
             [{source, "weight"}],
             fn [kernel] -> Nx.squeeze(kernel, axes: [1]) end
           }
         }}
      end
    end

    defp relative_bias_params_mapping() do
      %{
        "decoder.blocks.{n}.self_attention.relative_bias" => %{
          "kernel" => {
            [{"model.layers.{n}.self_attn.rel_logits_proj", "proj"}],
            fn [kernel] -> kernel end
          }
        }
      }
    end

    defp scale_params_mapping() do
      %{
        "decoder.blocks.{n}.ffn.scale" => %{
          "scale" => {
            [{"model.layers.{n}.mlp", "global_scale"}],
            fn [scale] -> Nx.reshape(scale, {1}) end
          }
        }
      }
    end

    defp expert_params_mapping(spec) do
      routed = "model.layers.{n}.mlp.experts"
      shared = "model.layers.{n}.mlp.shared_experts"

      %{
        "decoder.blocks.{n}.ffn.router" => %{
          "kernel" => {
            [{"model.layers.{n}.mlp.gate", "weight"}],
            fn [kernel] -> Nx.transpose(kernel) end
          },
          "score_correction_bias" => {
            [{"model.layers.{n}.mlp.gate", "e_score_correction_bias"}],
            fn [bias] -> bias end
          },
          "scale" => {
            [{"model.layers.{n}.mlp.gate", "global_scale"}],
            fn [scale] -> Nx.reshape(scale, {1}) end
          }
        },
        "decoder.blocks.{n}.ffn.experts" =>
          expert_kernels(routed, spec.num_experts, spec.moe_intermediate_size, true),
        "decoder.blocks.{n}.ffn.shared_experts" =>
          expert_kernels(shared, spec.num_shared_experts, nil, false)
      }
    end

    defp expert_kernels(python_layer_name, num_experts, intermediate_size, packed_gate_up?) do
      refs = fn projection ->
        for idx <- 0..(num_experts - 1) do
          [
            {join(python_layer_name, "#{idx}.#{projection}"), "weight"},
            {python_layer_name, projection}
          ]
        end
      end

      packed_refs = fn projection, packed ->
        for idx <- 0..(num_experts - 1) do
          [
            {join(python_layer_name, "#{idx}.#{projection}"), "weight"},
            {python_layer_name, packed}
          ]
        end
      end

      if packed_gate_up? do
        %{
          "gate_kernel" =>
            {packed_refs.("gate_proj", "gate_up_proj"),
             &Moe.stack_expert_kernels(&1, 0, intermediate_size)},
          "up_kernel" =>
            {packed_refs.("up_proj", "gate_up_proj"),
             &Moe.stack_expert_kernels(&1, intermediate_size, intermediate_size)},
          "down_kernel" => {packed_refs.("down_proj", "down_proj"), &Moe.stack_expert_kernels(&1)}
        }
      else
        %{
          "gate_kernel" => {refs.("gate_proj"), &Moe.stack_expert_kernels(&1)},
          "up_kernel" => {refs.("up_proj"), &Moe.stack_expert_kernels(&1)},
          "down_kernel" => {refs.("down_proj"), &Moe.stack_expert_kernels(&1)}
        }
      end
    end
  end
end
