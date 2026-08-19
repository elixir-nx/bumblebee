defmodule Bumblebee.Text.DeepseekV4 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 129_280,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 1_048_576,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process
        """
      ],
      hidden_size: [
        default: 4096,
        doc: "the dimensionality of hidden layers"
      ],
      moe_intermediate_size: [
        default: 2048,
        doc: "the dimensionality of intermediate layers in each expert"
      ],
      num_blocks: [
        default: 43,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 64,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      attention_head_size: [
        default: 512,
        doc: "the size of the query, key and value projection per attention head"
      ],
      query_lora_rank: [
        default: 1024,
        doc: "the rank of the low-rank projection for queries"
      ],
      partial_rotary_factor: [
        default: 0.25,
        doc: "the fraction of each head that rotary embedding is applied to"
      ],
      attention_window_size: [
        default: 128,
        doc: "the size of the sliding attention window"
      ],
      block_types: [
        default: nil,
        doc: """
        a list with the attention type of each block, one of `:sliding_attention`,
        `:compressed_sparse_attention` or `:heavily_compressed_attention`. When `nil`, the
        first two blocks use heavily compressed attention and the remaining ones alternate
        """
      ],
      compress_rate: [
        default: 4,
        doc: "the number of tokens compressed into a single entry in the sparse attention blocks"
      ],
      heavy_compress_rate: [
        default: 128,
        doc: """
        the number of tokens compressed into a single entry in the heavily compressed attention
        blocks
        """
      ],
      index_top_k: [
        default: 512,
        doc: "the number of compressed entries that the indexer selects for each query"
      ],
      index_num_heads: [
        default: 64,
        doc: "the number of scoring heads in the sparse attention indexer"
      ],
      index_head_size: [
        default: 128,
        doc: "the size of query and key projections per head in the sparse attention indexer"
      ],
      output_groups: [
        default: 8,
        doc: "the number of head groups in the grouped output projection"
      ],
      output_lora_rank: [
        default: 1024,
        doc: "the intermediate dimensionality per group in the grouped output projection"
      ],
      hyper_connection_multiplier: [
        default: 4,
        doc: """
        the number of parallel residual streams kept by the manifold-constrained hyper
        connections
        """
      ],
      sinkhorn_iterations: [
        default: 20,
        doc: """
        the number of Sinkhorn-Knopp iterations used to project the residual mixing matrix onto
        the doubly-stochastic manifold
        """
      ],
      hyper_connection_epsilon: [
        default: 1.0e-6,
        doc: "the numerical floor used by the hyper connections"
      ],
      mlp_block_types: [
        default: nil,
        doc: """
        a list with the mixture-of-experts type of each block, either `:moe` for the regular
        routed block, or `:hash_moe` for blocks where the experts are selected by a fixed
        token-to-expert lookup. When `nil`, the first three blocks use hash routing
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
        default: 1,
        doc: "the number of shared experts, which are applied to all tokens"
      ],
      routed_scaling_factor: [
        default: 1.5,
        doc: "the constant that the routing weights are multiplied by"
      ],
      swiglu_limit: [
        default: 10.0,
        doc: "the value that the expert gate and up projections are clamped to"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency in the sliding attention blocks"
      ],
      compress_rotary_embedding_base: [
        default: 160_000,
        doc: "base for computing rotary embedding frequency in the compressed attention blocks"
      ],
      compress_rotary_embedding_scaling_strategy: [
        default: nil,
        doc: "scaling configuration for the rotary embedding of the compressed attention blocks"
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
      tie_word_embeddings: [
        default: false,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: nil)

  @moduledoc """
  DeepSeek V4 model family.

  DeepSeek V4 departs from the usual Transformer design in several ways:

    * the residual is a stack of parallel streams, mixed in and out of
      every sublayer by manifold-constrained hyper connections (mHC),
      where the mixing matrix is projected onto the doubly-stochastic
      manifold with Sinkhorn-Knopp iterations

    * attention is shared-key-value multi-query attention over a sliding
      window, extended with compressed entries. Every `:compress_rate`
      tokens are pooled into a single entry, and in the compressed sparse
      attention blocks a lightweight indexer selects the highest scoring
      `:index_top_k` of those entries for each query

    * the output projection is low-rank and grouped over attention heads

    * the leading blocks route tokens to experts with a fixed
      token-to-expert lookup, rather than a learnt router

  ## Architectures

    * `:base` - plain DeepSeek V4 without any head on top

    * `:for_causal_language_modeling` - DeepSeek V4 with a language
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
    block_types = block_types(spec)

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      # Key and value are the same tensor and there is a single head
      decoder_num_attention_heads: 1,
      attention_head_size: spec.attention_head_size,
      decoder_num_blocks: spec.num_blocks,
      extra_states: fn idx ->
        case Enum.at(block_types, idx) do
          :sliding_attention ->
            []

          :heavily_compressed_attention ->
            [compressor: {2 * spec.attention_head_size}]

          :compressed_sparse_attention ->
            [
              compressor: {4 * spec.attention_head_size},
              indexer: {4 * spec.index_head_size}
            ]
        end
      end
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
    embeddings =
      Layers.default inputs["input_embeddings"] do
        Axon.embedding(inputs["input_ids"], spec.vocab_size, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: "embedder.token_embedding"
        )
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      decoder(
        embeddings,
        inputs["input_ids"],
        position_ids,
        inputs["attention_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      decoder_outputs.hidden_state
      |> hyper_connection_head(spec, name: "output_hyper_connection")
      |> Layers.rms_norm(name: "output_norm", epsilon: spec.layer_norm_epsilon)

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp decoder(hidden_state, input_ids, position_ids, attention_mask, cache, spec, opts) do
    name = opts[:name]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    block_types = block_types(spec)
    mlp_block_types = mlp_block_types(spec)

    # The residual is a stack of parallel streams
    streams =
      Axon.nx(hidden_state, fn hidden_state ->
        hidden_state
        |> Nx.new_axis(2)
        |> Nx.broadcast(
          put_elem(
            Tuple.insert_at(Nx.shape(hidden_state), 2, spec.hyper_connection_multiplier),
            2,
            spec.hyper_connection_multiplier
          )
        )
      end)

    state = %{
      hidden_state: streams,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cache: cache
    }

    outputs =
      for idx <- 0..(spec.num_blocks - 1), reduce: state do
        state ->
          block_name = join(join(name, "blocks"), idx)
          block_type = Enum.at(block_types, idx)

          block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

          {attention_cache, cross_attention_cache} =
            Layers.Decoder.get_attention_caches(block_cache)

          {post, comb, collapsed} =
            hyper_connection(state.hidden_state, spec,
              name: join(block_name, "self_attention_hc")
            )

          {attention_output, attention, attention_cache} =
            collapsed
            |> Layers.rms_norm(
              name: join(block_name, "self_attention_norm"),
              epsilon: spec.layer_norm_epsilon
            )
            |> attention(
              position_ids,
              attention_mask,
              attention_cache,
              offset,
              block_type,
              spec,
              name: join(block_name, "self_attention")
            )

          streams = mix_streams(state.hidden_state, attention_output, post, comb)

          {post, comb, collapsed} =
            hyper_connection(streams, spec, name: join(block_name, "ffn_hc"))

          ffn_output =
            collapsed
            |> Layers.rms_norm(
              name: join(block_name, "output_norm"),
              epsilon: spec.layer_norm_epsilon
            )
            |> ffn(input_ids, Enum.at(mlp_block_types, idx), spec, name: join(block_name, "ffn"))

          streams = mix_streams(streams, ffn_output, post, comb)

          block_cache =
            Layers.Decoder.put_attention_caches(
              block_cache,
              attention_cache,
              cross_attention_cache
            )

          %{
            hidden_state: streams,
            hidden_states: Layers.append(state.hidden_states, streams),
            attentions: Layers.append(state.attentions, attention),
            cache: Layers.Decoder.put_block_cache(state.cache, idx, block_cache)
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  # Places the sublayer output into the residual streams and mixes the
  # streams with the doubly-stochastic combination matrix
  defp mix_streams(streams, output, post, comb) do
    Axon.layer(&mix_streams_impl/5, [streams, output, post, comb])
  end

  defnp mix_streams_impl(streams, output, post, comb, _opts \\ []) do
    type = Nx.type(streams)

    placed = Nx.new_axis(Nx.as_type(post, type), -1) * Nx.new_axis(output, -2)

    comb = Nx.as_type(comb, type)

    # sum_j comb[..., j, k] * streams[..., j, :]
    mixed = Nx.dot(comb, [2], [0, 1], streams, [2], [0, 1])

    placed + mixed
  end

  # Manifold-constrained hyper connections. Returns the weights used to
  # place the sublayer output back into the streams, the stream mixing
  # matrix and the collapsed input for the sublayer
  defp hyper_connection(streams, spec, opts) do
    name = opts[:name]

    multiplier = spec.hyper_connection_multiplier
    mix = (2 + multiplier) * multiplier

    kernel =
      Axon.param("kernel", fn _ -> {mix, multiplier * spec.hidden_size} end,
        initializer: kernel_initializer(spec)
      )

    base = Axon.param("base", fn _ -> {mix} end, initializer: :zeros)
    scale = Axon.param("scale", fn _ -> {3} end, initializer: :ones)

    Axon.layer(&hyper_connection_impl/5, [streams, kernel, base, scale],
      name: name,
      op_name: :hyper_connection,
      multiplier: multiplier,
      epsilon: spec.hyper_connection_epsilon,
      norm_epsilon: spec.layer_norm_epsilon,
      sinkhorn_iterations: spec.sinkhorn_iterations
    )
    |> Layers.unwrap_tuple(3)
  end

  defnp hyper_connection_impl(streams, kernel, base, scale, opts \\ []) do
    opts =
      keyword!(opts, [
        :multiplier,
        :epsilon,
        :norm_epsilon,
        :sinkhorn_iterations,
        mode: :inference
      ])

    multiplier = opts[:multiplier]
    epsilon = opts[:epsilon]

    {batch_size, sequence_length, _multiplier, hidden_size} = Nx.shape(streams)

    flat =
      streams
      |> Nx.reshape({batch_size, sequence_length, multiplier * hidden_size})
      |> Nx.as_type(:f32)
      |> unweighted_rms_norm(opts[:norm_epsilon])

    mixes = Nx.dot(flat, [-1], Nx.as_type(kernel, :f32), [1])

    base = Nx.as_type(base, :f32)
    scale = Nx.as_type(scale, :f32)

    pre_weights = Nx.slice_along_axis(mixes, 0, multiplier, axis: -1)
    post_weights = Nx.slice_along_axis(mixes, multiplier, multiplier, axis: -1)
    comb_weights = Nx.slice_along_axis(mixes, 2 * multiplier, multiplier * multiplier, axis: -1)

    pre_base = Nx.slice_along_axis(base, 0, multiplier, axis: 0)
    post_base = Nx.slice_along_axis(base, multiplier, multiplier, axis: 0)

    comb_base =
      base
      |> Nx.slice_along_axis(2 * multiplier, multiplier * multiplier, axis: 0)
      |> Nx.reshape({multiplier, multiplier})

    pre = Nx.sigmoid(pre_weights * scale[0] + pre_base) + epsilon
    post = 2 * Nx.sigmoid(post_weights * scale[1] + post_base)

    comb_logits =
      comb_weights
      |> Nx.reshape({batch_size, sequence_length, multiplier, multiplier})
      |> Nx.multiply(scale[2])
      |> Nx.add(comb_base)

    comb = Axon.Activations.softmax(comb_logits, axis: -1) + epsilon
    comb = comb / (Nx.sum(comb, axes: [-2], keep_axes: true) + epsilon)
    comb = sinkhorn(comb, opts[:sinkhorn_iterations], epsilon)

    collapsed =
      (Nx.new_axis(pre, -1) * Nx.as_type(streams, :f32))
      |> Nx.sum(axes: [2])
      |> Nx.as_type(Nx.type(streams))

    {post, comb, collapsed}
  end

  deftransformp sinkhorn(comb, iterations, epsilon) do
    for _ <- 1..(iterations - 1)//1, reduce: comb do
      comb ->
        comb = Nx.divide(comb, Nx.add(Nx.sum(comb, axes: [-1], keep_axes: true), epsilon))
        Nx.divide(comb, Nx.add(Nx.sum(comb, axes: [-2], keep_axes: true), epsilon))
    end
  end

  # The final collapse of the residual streams
  defp hyper_connection_head(streams, spec, opts) do
    name = opts[:name]

    multiplier = spec.hyper_connection_multiplier

    kernel =
      Axon.param("kernel", fn _ -> {multiplier, multiplier * spec.hidden_size} end,
        initializer: kernel_initializer(spec)
      )

    base = Axon.param("base", fn _ -> {multiplier} end, initializer: :zeros)
    scale = Axon.param("scale", fn _ -> {1} end, initializer: :ones)

    Axon.layer(&hyper_connection_head_impl/5, [streams, kernel, base, scale],
      name: name,
      op_name: :hyper_connection_head,
      multiplier: multiplier,
      epsilon: spec.hyper_connection_epsilon,
      norm_epsilon: spec.layer_norm_epsilon
    )
  end

  defnp hyper_connection_head_impl(streams, kernel, base, scale, opts \\ []) do
    opts = keyword!(opts, [:multiplier, :epsilon, :norm_epsilon, mode: :inference])

    {batch_size, sequence_length, multiplier, hidden_size} = Nx.shape(streams)

    flat =
      streams
      |> Nx.reshape({batch_size, sequence_length, multiplier * hidden_size})
      |> Nx.as_type(:f32)
      |> unweighted_rms_norm(opts[:norm_epsilon])

    mixes = Nx.dot(flat, [-1], Nx.as_type(kernel, :f32), [1])

    pre =
      Nx.sigmoid(mixes * Nx.as_type(scale, :f32) + Nx.as_type(base, :f32)) + opts[:epsilon]

    (Nx.new_axis(pre, -1) * Nx.as_type(streams, :f32))
    |> Nx.sum(axes: [2])
    |> Nx.as_type(Nx.type(streams))
  end

  defnp unweighted_rms_norm(input, epsilon) do
    input * Nx.rsqrt(Nx.mean(Nx.pow(input, 2), axes: [-1], keep_axes: true) + epsilon)
  end

  defp attention(
         hidden_state,
         position_ids,
         attention_mask,
         attention_cache,
         offset,
         block_type,
         spec,
         opts
       ) do
    name = opts[:name]

    num_heads = spec.num_attention_heads
    head_size = spec.attention_head_size
    rotary_size = rotary_size(spec)

    {rotary_base, scaling_strategy} = rotary_config(spec, block_type)

    query_latent =
      hidden_state
      |> Axon.dense(spec.query_lora_rank,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query_down"),
        use_bias: false
      )
      |> Layers.rms_norm(
        name: join(name, "query_norm"),
        epsilon: spec.layer_norm_epsilon
      )

    query =
      query_latent
      |> Axon.dense(num_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query_up"),
        use_bias: false
      )
      |> Layers.split_heads(num_heads)
      |> then(&scaleless_rms_norm(&1, spec.layer_norm_epsilon))
      |> rotate(position_ids, offset,
        size: rotary_size,
        base: rotary_base,
        scaling_strategy: scaling_strategy,
        max_positions: spec.max_positions
      )

    # Key and value are the same single-head tensor
    key_value =
      hidden_state
      |> Axon.dense(head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key_value"),
        use_bias: false
      )
      |> Layers.rms_norm(
        name: join(name, "key_value_norm"),
        epsilon: spec.layer_norm_epsilon
      )
      |> Layers.split_heads(1)
      |> rotate(position_ids, offset,
        size: rotary_size,
        base: rotary_base,
        scaling_strategy: scaling_strategy,
        max_positions: spec.max_positions
      )

    {key_value, _value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(
        key_value,
        key_value,
        attention_cache,
        offset
      )

    {compressed, block_bias, attention_cache} =
      case block_type do
        :sliding_attention ->
          {nil, nil, attention_cache}

        _other ->
          compressor(
            hidden_state,
            query_latent,
            position_ids,
            attention_mask,
            attention_cache,
            offset,
            block_type,
            spec,
            name: join(name, "compressor")
          )
      end

    {key_value, bias} =
      combined_key_values(
        query,
        key_value,
        compressed,
        block_bias,
        attention_mask,
        offset,
        spec.attention_window_size
      )

    sinks =
      Axon.layer(
        fn _hidden_state, sinks, _opts -> sinks end,
        [hidden_state, Axon.param("sinks", fn _ -> {num_heads} end, initializer: :zeros)],
        name: join(name, "sinks"),
        op_name: :attention_sinks
      )

    key_value = Layers.repeat_interleave(key_value, num_heads, axis: 2)

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key_value,
        key_value,
        Layers.none(),
        Layers.none(),
        bias,
        offset,
        causal: false,
        scale: 1.0 / :math.sqrt(head_size),
        sinks: sinks
      )

    # Key and value are the same tensor, so the value carried the rotary
    # embedding into the output. We rotate it back by the query position,
    # which leaves the contribution of each entry a function of the
    # relative distance only
    attention_output =
      attention_output
      |> rotate(position_ids, offset,
        size: rotary_size,
        base: rotary_base,
        scaling_strategy: scaling_strategy,
        max_positions: spec.max_positions,
        negate: true
      )
      |> grouped_output_projection(spec, name: name)

    {attention_output, attention_weights, attention_cache}
  end

  # Combines the sliding window mask over the regular entries with the
  # per-query bias over the compressed entries
  defp combined_key_values(
         query,
         key_value,
         compressed,
         block_bias,
         attention_mask,
         offset,
         window_size
       ) do
    bias =
      Axon.layer(
        &sliding_bias_impl/5,
        [query, key_value, Axon.optional(attention_mask), Axon.optional(offset)],
        window_size: window_size
      )

    if compressed do
      {Axon.concatenate([key_value, compressed], axis: 1),
       Axon.concatenate([bias, block_bias], axis: -1)}
    else
      {key_value, bias}
    end
  end

  defnp sliding_bias_impl(query, key_value, attention_mask, offset, opts \\ []) do
    opts = keyword!(opts, [:window_size, mode: :inference])

    key_length = Nx.axis_size(key_value, 1)
    query_length = Nx.axis_size(query, 1)

    offset =
      case offset do
        %Axon.None{} -> 0
        offset -> offset
      end

    distance = Nx.iota({query_length, 1}) + offset - Nx.iota({1, key_length})

    mask = Nx.logical_and(Nx.greater_equal(distance, 0), Nx.less(distance, opts[:window_size]))

    mask =
      case attention_mask do
        %Axon.None{} ->
          Nx.new_axis(mask, 0)

        attention_mask ->
          Nx.logical_and(Nx.new_axis(mask, 0), Nx.new_axis(attention_mask, 1))
      end

    mask
    |> Nx.select(Nx.tensor(0.0, type: :f32), Nx.Constants.min_finite(:f32))
    |> Nx.new_axis(1)
  end

  defp grouped_output_projection(attention_output, spec, opts) do
    name = opts[:name]

    groups = spec.output_groups
    inner_size = spec.num_attention_heads * spec.attention_head_size

    kernel =
      Axon.param(
        "kernel",
        fn _ -> {groups, div(inner_size, groups), spec.output_lora_rank} end,
        initializer: kernel_initializer(spec)
      )

    flat = Layers.flatten_trailing(attention_output)

    grouped =
      Axon.layer(&grouped_projection_impl/3, [flat, kernel],
        name: join(name, "output_down"),
        op_name: :grouped_dense,
        groups: groups
      )

    Axon.dense(grouped, spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output_up"),
      use_bias: false
    )
  end

  defnp grouped_projection_impl(hidden_state, kernel, opts \\ []) do
    opts = keyword!(opts, [:groups, mode: :inference])

    {batch_size, sequence_length, size} = Nx.shape(hidden_state)
    groups = opts[:groups]

    hidden_state
    |> Nx.reshape({batch_size * sequence_length, groups, div(size, groups)})
    |> Nx.transpose(axes: [1, 0, 2])
    |> Nx.dot([2], [0], kernel, [1], [0])
    |> Nx.transpose(axes: [1, 0, 2])
    |> Nx.reshape({batch_size, sequence_length, :auto})
  end

  # Pools every `compress_rate` tokens into a single entry. In the
  # compressed sparse attention blocks, an indexer then selects the
  # highest scoring entries for each query
  defp compressor(
         hidden_state,
         query_latent,
         position_ids,
         attention_mask,
         attention_cache,
         offset,
         block_type,
         spec,
         opts
       ) do
    name = opts[:name]

    head_size = spec.attention_head_size
    overlapping? = block_type == :compressed_sparse_attention

    rate =
      if overlapping?, do: spec.compress_rate, else: spec.heavy_compress_rate

    {compressed, attention_cache} =
      compress(
        hidden_state,
        attention_cache,
        offset,
        :compressor,
        head_size,
        rate,
        overlapping?,
        spec,
        name: name
      )

    compressed =
      rotate_windows(compressed, rate, spec, size: rotary_size(spec))

    {block_bias, attention_cache} =
      if overlapping? do
        indexer(
          hidden_state,
          query_latent,
          position_ids,
          attention_mask,
          attention_cache,
          offset,
          rate,
          spec,
          name: join(name, "indexer")
        )
      else
        {window_bias(compressed, position_ids, offset, rate), attention_cache}
      end

    {Axon.nx(compressed, & &1), block_bias, attention_cache}
  end

  # Projects every token into a key-value entry and a gate, then pools
  # the tokens of each window with a softmax over the gates
  defp compress(
         hidden_state,
         attention_cache,
         offset,
         cache_name,
         head_size,
         rate,
         overlapping?,
         spec,
         opts
       ) do
    name = opts[:name]

    factor = if overlapping?, do: 2, else: 1

    key_value =
      Axon.dense(hidden_state, factor * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key_value"),
        use_bias: false
      )

    gate =
      Axon.dense(hidden_state, factor * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "gate"),
        use_bias: false
      )

    projections = Axon.concatenate([key_value, gate], axis: -1)

    {projections, attention_cache} =
      Layers.Decoder.cached_state(projections, cache_name, attention_cache, offset)

    position_bias =
      Axon.param("position_bias", fn _ -> {rate, factor * head_size} end, initializer: :zeros)

    weight =
      Axon.param("weight", fn _ -> {head_size} end, initializer: :ones)

    compressed =
      Axon.layer(&compress_impl/4, [projections, position_bias, weight],
        name: join(name, "compress"),
        op_name: :window_compression,
        rate: rate,
        head_size: head_size,
        overlapping: overlapping?,
        epsilon: spec.layer_norm_epsilon
      )

    {compressed, attention_cache}
  end

  defnp compress_impl(projections, position_bias, weight, opts \\ []) do
    opts =
      keyword!(opts, [:rate, :head_size, :overlapping, :epsilon, mode: :inference])

    rate = opts[:rate]
    head_size = opts[:head_size]

    {batch_size, length, size} = Nx.shape(projections)
    half = div(size, 2)

    # Only complete windows are pooled, the trailing tokens are dropped.
    # We always keep at least one window, even when the sequence is
    # shorter, in which case it is entirely masked out downstream
    windows = max(div(length, rate), 1)

    projections = complete_windows(projections, rate)

    key_value =
      projections
      |> Nx.slice_along_axis(0, half, axis: -1)
      |> Nx.reshape({batch_size, windows, rate, half})

    gate =
      projections
      |> Nx.slice_along_axis(half, half, axis: -1)
      |> Nx.reshape({batch_size, windows, rate, half})
      |> Nx.add(position_bias)

    {key_value, gate} =
      case opts[:overlapping] do
        false ->
          {key_value, gate}

        true ->
          # Every token contributes to the current window and to the next
          # one, so each window pools twice as many slots
          current = Nx.slice_along_axis(key_value, head_size, head_size, axis: -1)
          previous = Nx.slice_along_axis(key_value, 0, head_size, axis: -1)

          current_gate = Nx.slice_along_axis(gate, head_size, head_size, axis: -1)
          previous_gate = Nx.slice_along_axis(gate, 0, head_size, axis: -1)

          previous = shift_windows(previous, 0.0)
          previous_gate = shift_windows(previous_gate, Nx.Constants.neg_infinity(:f32))

          {Nx.concatenate([previous, current], axis: 2),
           Nx.concatenate([previous_gate, current_gate], axis: 2)}
      end

    weights =
      gate
      |> Nx.as_type(:f32)
      |> Axon.Activations.softmax(axis: 2)
      |> Nx.as_type(Nx.type(key_value))

    pooled = Nx.sum(key_value * weights, axes: [2])

    normalized =
      pooled * Nx.rsqrt(Nx.mean(Nx.pow(pooled, 2), axes: [-1], keep_axes: true) + opts[:epsilon])

    Nx.new_axis(normalized * weight, 2)
  end

  # Shifts the windows by one, so that each window sees the previous
  # one's contribution. The first window has no predecessor
  deftransformp shift_windows(tensor, fill_value) do
    {batch_size, windows, rate, size} = Nx.shape(tensor)

    fill =
      fill_value
      |> Nx.tensor(type: Nx.type(tensor))
      |> Nx.broadcast({batch_size, 1, rate, size})

    if windows == 1 do
      fill
    else
      Nx.concatenate([fill, Nx.slice_along_axis(tensor, 0, windows - 1, axis: 1)], axis: 1)
    end
  end

  # Keeps only the tokens that form complete windows. When there are
  # none, a single zeroed window is kept, so that the shapes stay valid
  deftransformp complete_windows(projections, rate) do
    {batch_size, length, size} = Nx.shape(projections)
    usable = div(length, rate) * rate

    if usable == 0 do
      Nx.broadcast(Nx.tensor(0.0, type: Nx.type(projections)), {batch_size, rate, size})
    else
      Nx.slice_along_axis(projections, 0, usable, axis: 1)
    end
  end

  # The bias masking out the compressed entries that a query must not
  # attend to, that is, the ones pooling tokens at or after the query
  defp window_bias(compressed, position_ids, offset, rate) do
    Axon.layer(&window_bias_impl/4, [compressed, position_ids, Axon.optional(offset)], rate: rate)
  end

  defnp window_bias_impl(compressed, position_ids, offset, opts \\ []) do
    opts = keyword!(opts, [:rate, mode: :inference])

    windows = Nx.axis_size(compressed, 1)

    offset =
      case offset do
        %Axon.None{} -> 0
        offset -> offset
      end

    query_positions = Nx.iota({Nx.axis_size(position_ids, 1)}) + offset

    threshold = Nx.quotient(query_positions + 1, opts[:rate])

    visible = Nx.less(Nx.new_axis(Nx.iota({windows}), 0), Nx.new_axis(threshold, 1))

    visible
    |> Nx.select(Nx.tensor(0.0, type: :f32), Nx.Constants.min_finite(:f32))
    |> Nx.new_axis(0)
    |> Nx.new_axis(0)
  end

  # Scores the compressed entries against each query and keeps the
  # highest scoring ones
  defp indexer(
         hidden_state,
         query_latent,
         position_ids,
         _attention_mask,
         attention_cache,
         offset,
         rate,
         spec,
         opts
       ) do
    name = opts[:name]

    head_size = spec.index_head_size

    {compressed, attention_cache} =
      compress(
        hidden_state,
        attention_cache,
        offset,
        :indexer,
        head_size,
        rate,
        true,
        spec,
        name: name
      )

    compressed = rotate_windows(compressed, rate, spec, size: head_size)

    query =
      query_latent
      |> Axon.dense(spec.index_num_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query"),
        use_bias: false
      )
      |> Layers.split_heads(spec.index_num_heads)
      |> rotate(position_ids, offset,
        size: head_size,
        base: spec.compress_rotary_embedding_base,
        scaling_strategy: spec.compress_rotary_embedding_scaling_strategy,
        max_positions: spec.max_positions
      )

    head_weights =
      Axon.dense(hidden_state, spec.index_num_heads,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "head_weights"),
        use_bias: false
      )

    bias =
      Axon.layer(
        &indexer_bias_impl/5,
        [query, compressed, head_weights, Axon.optional(offset)],
        op_name: :sparse_attention_indexer,
        index_top_k: spec.index_top_k,
        rate: rate,
        scale: 1.0 / :math.sqrt(head_size),
        head_scale: 1.0 / :math.sqrt(spec.index_num_heads)
      )

    {bias, attention_cache}
  end

  defnp indexer_bias_impl(query, compressed, head_weights, offset, opts \\ []) do
    opts =
      keyword!(opts, [:index_top_k, :rate, :scale, :head_scale, mode: :inference])

    query = Nx.as_type(query, :f32)
    key = Nx.as_type(Nx.squeeze(compressed, axes: [2]), :f32)

    # {batch_size, sequence_length, num_heads, windows}
    scores = Nx.dot(query, [3], [0], key, [2], [0])
    scores = Axon.Activations.relu(scores) * opts[:scale]

    head_weights = Nx.as_type(head_weights, :f32) * opts[:head_scale]

    index_scores = Nx.sum(scores * Nx.new_axis(head_weights, -1), axes: [2])

    query_length = Nx.axis_size(query, 1)
    windows = Nx.axis_size(key, 1)

    offset =
      case offset do
        %Axon.None{} -> 0
        offset -> offset
      end

    threshold = Nx.quotient(Nx.iota({query_length, 1}) + offset + 1, opts[:rate])
    visible = Nx.less(Nx.iota({1, windows}), threshold)

    index_scores =
      Nx.select(Nx.new_axis(visible, 0), index_scores, Nx.Constants.neg_infinity(:f32))

    top_k = min(opts[:index_top_k], windows)

    order = Nx.argsort(index_scores, axis: -1, direction: :desc, stable: true)
    rank = Nx.argsort(order, axis: -1, stable: true)

    selected = Nx.logical_and(Nx.less(rank, top_k), Nx.new_axis(visible, 0))

    selected
    |> Nx.select(Nx.tensor(0.0, type: :f32), Nx.Constants.min_finite(:f32))
    |> Nx.new_axis(1)
  end

  # Applies rotary embedding to the trailing part of each head, keeping
  # the interleaved layout
  defp rotate(hidden_state, position_ids, offset, opts) do
    opts =
      Keyword.validate!(opts, [
        :size,
        :base,
        :scaling_strategy,
        :max_positions,
        negate: false
      ])

    Axon.layer(&rotate_impl/4, [hidden_state, position_ids, Axon.optional(offset)],
      op_name: :rotary_embedding,
      size: opts[:size],
      base: opts[:base],
      scaling_strategy: opts[:scaling_strategy],
      negate: opts[:negate]
    )
  end

  defnp rotate_impl(hidden_state, _position_ids, offset, opts \\ []) do
    opts = keyword!(opts, [:size, :base, :scaling_strategy, :negate, mode: :inference])

    offset =
      case offset do
        %Axon.None{} -> 0
        offset -> offset
      end

    positions = Nx.iota({Nx.axis_size(hidden_state, 1)}) + offset

    {cos, sin} = rotary_frequencies(positions, opts[:size], opts[:base], opts[:scaling_strategy])

    sin = if opts[:negate], do: -sin, else: sin

    apply_interleaved_rotary(hidden_state, cos, sin)
  end

  # Rotates the compressed entries at the position of the first token of
  # each window
  defp rotate_windows(compressed, rate, spec, opts) do
    Axon.layer(&rotate_windows_impl/2, [compressed],
      op_name: :rotary_embedding,
      size: opts[:size],
      rate: rate,
      base: spec.compress_rotary_embedding_base,
      scaling_strategy: spec.compress_rotary_embedding_scaling_strategy
    )
  end

  defnp rotate_windows_impl(compressed, opts \\ []) do
    opts = keyword!(opts, [:size, :rate, :base, :scaling_strategy, mode: :inference])

    positions = Nx.iota({Nx.axis_size(compressed, 1)}) * opts[:rate]

    {cos, sin} = rotary_frequencies(positions, opts[:size], opts[:base], opts[:scaling_strategy])

    apply_interleaved_rotary(compressed, cos, sin)
  end

  defnp apply_interleaved_rotary(hidden_state, cos, sin) do
    # {sequence_length, size / 2} -> {1, sequence_length, 1, size}
    cos = cos |> repeat_pairs() |> Nx.new_axis(0) |> Nx.new_axis(2)
    sin = sin |> repeat_pairs() |> Nx.new_axis(0) |> Nx.new_axis(2)

    size = Nx.axis_size(cos, -1)
    head_size = Nx.axis_size(hidden_state, -1)

    rotary =
      hidden_state
      |> Nx.slice_along_axis(head_size - size, size, axis: -1)
      |> Nx.as_type(:f32)

    even = rotary[[.., .., .., 0..-1//2]]
    odd = rotary[[.., .., .., 1..-1//2]]

    cos_half = cos[[.., .., .., 0..-1//2]]
    sin_half = sin[[.., .., .., 0..-1//2]]

    rotated =
      [
        Nx.new_axis(even * cos_half - odd * sin_half, -1),
        Nx.new_axis(odd * cos_half + even * sin_half, -1)
      ]
      |> Nx.concatenate(axis: -1)
      |> Nx.reshape(Nx.shape(rotary))
      |> Nx.as_type(Nx.type(hidden_state))

    prepend_pass(hidden_state, rotated, head_size - size)
  end

  # Only the trailing part of each head is rotated
  deftransformp prepend_pass(hidden_state, rotated, pass_size) do
    if pass_size == 0 do
      rotated
    else
      pass = Nx.slice_along_axis(hidden_state, 0, pass_size, axis: -1)
      Nx.concatenate([pass, rotated], axis: -1)
    end
  end

  defnp repeat_pairs(tensor) do
    {length, size} = Nx.shape(tensor)

    tensor
    |> Nx.new_axis(-1)
    |> Nx.broadcast({length, size, 2})
    |> Nx.reshape({length, 2 * size})
  end

  deftransformp rotary_frequencies(positions, size, base, scaling_strategy) do
    range = Nx.iota({div(size, 2)}) |> Nx.multiply(2) |> Nx.divide(size)

    inv_frequency =
      case scaling_strategy do
        %{type: :yarn} = strategy ->
          yarn_inv_frequency(base, size, strategy)

        _other ->
          Nx.divide(1.0, Nx.pow(base, range))
      end

    angle = Nx.outer(Nx.as_type(positions, :f32), inv_frequency)

    {Nx.cos(angle), Nx.sin(angle)}
  end

  # YaRN interpolates between the original and the scaled frequencies.
  # DeepSeek V4 does not apply the attention factor to cos/sin
  deftransformp yarn_inv_frequency(base, size, strategy) do
    %{
      factor: factor,
      original_max_positions: original_max_positions,
      beta_fast: beta_fast,
      beta_slow: beta_slow
    } = strategy

    correction_dim = fn rotations ->
      size * :math.log(original_max_positions / (rotations * 2 * :math.pi())) /
        (2 * :math.log(base))
    end

    low = max(Float.floor(correction_dim.(beta_fast)), 0)
    high = min(Float.ceil(correction_dim.(beta_slow)), size - 1)
    high = if low == high, do: high + 0.001, else: high

    range = Nx.iota({div(size, 2)}) |> Nx.multiply(2) |> Nx.divide(size)
    positional_frequency = Nx.pow(base, range)

    extrapolation_factor =
      Nx.iota({div(size, 2)})
      |> Nx.subtract(low)
      |> Nx.divide(high - low)
      |> Nx.clip(0, 1)
      |> then(&Nx.subtract(1, &1))

    interpolation = Nx.divide(1.0, Nx.multiply(factor, positional_frequency))
    extrapolation = Nx.divide(1.0, positional_frequency)

    Nx.add(
      Nx.multiply(interpolation, Nx.subtract(1, extrapolation_factor)),
      Nx.multiply(extrapolation, extrapolation_factor)
    )
  end

  defp scaleless_rms_norm(hidden_state, epsilon) do
    Axon.layer(&scaleless_rms_norm_impl/2, [hidden_state], epsilon: epsilon)
  end

  defnp scaleless_rms_norm_impl(hidden_state, opts \\ []) do
    opts = keyword!(opts, [:epsilon, mode: :inference])

    type = Nx.type(hidden_state)

    hidden_state
    |> Nx.as_type(:f32)
    |> unweighted_rms_norm(opts[:epsilon])
    |> Nx.as_type(type)
  end

  defp ffn(hidden_state, input_ids, mlp_block_type, spec, opts) do
    name = opts[:name]

    weights =
      case mlp_block_type do
        :moe ->
          Moe.router(hidden_state,
            num_experts: spec.num_experts,
            num_experts_per_token: spec.num_experts_per_token,
            scoring: :sqrt_softplus,
            normalize_top_k: true,
            routed_scaling_factor: spec.routed_scaling_factor,
            use_score_correction_bias: true,
            kernel_initializer: kernel_initializer(spec),
            name: join(name, "router")
          )

        :hash_moe ->
          hash_router(hidden_state, input_ids, spec, name: join(name, "router"))
      end

    routed =
      Moe.experts(hidden_state, weights,
        num_experts: spec.num_experts,
        hidden_size: spec.hidden_size,
        intermediate_size: spec.moe_intermediate_size,
        activation: spec.activation,
        variant: :clamped_gated,
        clamp_limit: spec.swiglu_limit,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "experts")
      )

    shared =
      clamped_gated_ffn(hidden_state, spec, name: join(name, "shared_expert"))

    Axon.add(routed, shared)
  end

  # The experts are selected by a fixed token-to-expert lookup, only
  # their weights are learnt
  defp hash_router(hidden_state, input_ids, spec, opts) do
    name = opts[:name]

    kernel =
      Axon.param(
        "kernel",
        fn shape, _ -> {elem(shape, tuple_size(shape) - 1), spec.num_experts} end,
        initializer: kernel_initializer(spec)
      )

    token_experts =
      Axon.param(
        "token_experts",
        fn _, _ -> {spec.vocab_size, spec.num_experts_per_token} end,
        initializer: :zeros
      )

    Axon.layer(&hash_router_impl/5, [hidden_state, input_ids, kernel, token_experts],
      name: name,
      op_name: :moe_hash_router,
      num_experts: spec.num_experts,
      routed_scaling_factor: spec.routed_scaling_factor
    )
  end

  defnp hash_router_impl(hidden_state, input_ids, kernel, token_experts, opts \\ []) do
    opts = keyword!(opts, [:num_experts, :routed_scaling_factor, mode: :inference])

    {batch_size, sequence_length, _hidden_size} = Nx.shape(hidden_state)

    hidden_state = Nx.as_type(hidden_state, :f32)
    logits = Nx.dot(hidden_state, [-1], Nx.as_type(kernel, :f32), [0])
    scores = Nx.sqrt(Axon.Activations.softplus(logits))

    # {batch_size, sequence_length, num_experts_per_token}
    indices =
      token_experts
      |> Nx.take(Nx.as_type(input_ids, :s64))
      |> Nx.as_type(:s64)

    mask =
      indices
      |> Nx.reshape({batch_size, sequence_length, :auto, 1})
      |> Nx.equal(Nx.iota({1, 1, 1, opts[:num_experts]}))
      |> Nx.sum(axes: [2])
      |> Nx.min(1)
      |> Nx.as_type(:f32)

    weights = scores * mask
    weights = weights / (Nx.sum(weights, axes: [-1], keep_axes: true) + 1.0e-20)

    weights * opts[:routed_scaling_factor]
  end

  defp clamped_gated_ffn(hidden_state, spec, opts) do
    name = opts[:name]

    gate =
      Axon.dense(hidden_state, spec.moe_intermediate_size,
        name: join(name, "gate"),
        use_bias: false
      )

    intermediate =
      Axon.dense(hidden_state, spec.moe_intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    hidden_state =
      Axon.layer(&clamped_gate_impl/3, [gate, intermediate],
        limit: spec.swiglu_limit,
        activation: spec.activation
      )

    Axon.dense(hidden_state, spec.hidden_size, name: join(name, "output"), use_bias: false)
  end

  defnp clamped_gate_impl(gate, intermediate, opts \\ []) do
    opts = keyword!(opts, [:limit, :activation, mode: :inference])

    limit = opts[:limit]

    gate = Nx.min(gate, limit)
    intermediate = Nx.clip(intermediate, -limit, limit)

    Axon.Activations.silu(gate) * intermediate
  end

  defp rotary_size(spec), do: trunc(spec.attention_head_size * spec.partial_rotary_factor)

  defp rotary_config(spec, :sliding_attention), do: {spec.rotary_embedding_base, nil}

  defp rotary_config(spec, _other),
    do: {spec.compress_rotary_embedding_base, spec.compress_rotary_embedding_scaling_strategy}

  defp block_types(spec) do
    spec.block_types ||
      List.duplicate(:heavily_compressed_attention, min(spec.num_blocks, 2)) ++
        for idx <- 0..(max(spec.num_blocks - 2, 0) - 1)//1 do
          if rem(idx, 2) == 1,
            do: :compressed_sparse_attention,
            else: :heavily_compressed_attention
        end
  end

  defp mlp_block_types(spec) do
    spec.mlp_block_types ||
      List.duplicate(:hash_moe, min(spec.num_blocks, 3)) ++
        List.duplicate(:moe, max(spec.num_blocks - 3, 0))
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      block_type_converter =
        list(
          mapping(%{
            "sliding_attention" => :sliding_attention,
            "compressed_sparse_attention" => :compressed_sparse_attention,
            "heavily_compressed_attention" => :heavily_compressed_attention
          })
        )

      mlp_block_type_converter = list(mapping(%{"moe" => :moe, "hash_moe" => :hash_moe}))

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          moe_intermediate_size: {"moe_intermediate_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          attention_head_size: {"head_dim", number()},
          query_lora_rank: {"q_lora_rank", number()},
          partial_rotary_factor: {"partial_rotary_factor", number()},
          attention_window_size: {"sliding_window", number()},
          block_types: {"layer_types", optional(block_type_converter)},
          mlp_block_types: {"mlp_layer_types", optional(mlp_block_type_converter)},
          index_top_k: {"index_topk", number()},
          index_num_heads: {"index_n_heads", number()},
          index_head_size: {"index_head_dim", number()},
          output_groups: {"o_groups", number()},
          output_lora_rank: {"o_lora_rank", number()},
          hyper_connection_multiplier: {"hc_mult", number()},
          sinkhorn_iterations: {"hc_sinkhorn_iters", number()},
          hyper_connection_epsilon: {"hc_eps", number()},
          num_experts: {"n_routed_experts", number()},
          num_experts_per_token: {"num_experts_per_tok", number()},
          num_shared_experts: {"n_shared_experts", number()},
          routed_scaling_factor: {"routed_scaling_factor", number()},
          swiglu_limit: {"swiglu_limit", number()},
          activation: {"hidden_act", activation()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      opts = Keyword.merge(opts, compress_options(data))
      opts = Keyword.merge(opts, rotary_options(data))

      @for.config(spec, opts)
    end

    defp compress_options(data) do
      case data["compress_rates"] do
        %{} = rates ->
          []
          |> put_option(:compress_rate, rates["compressed_sparse_attention"])
          |> put_option(:heavy_compress_rate, rates["heavily_compressed_attention"])

        _other ->
          []
          |> put_option(:compress_rate, data["compress_rate_csa"])
          |> put_option(:heavy_compress_rate, data["compress_rate_hca"])
      end
    end

    defp rotary_options(data) do
      opts = put_option([], :compress_rotary_embedding_base, data["compress_rope_theta"])

      case data["rope_parameters"] do
        %{"main" => %{} = main, "compress" => %{} = compress} ->
          opts
          |> put_option(:rotary_embedding_base, main["rope_theta"])
          |> put_option(:compress_rotary_embedding_base, compress["rope_theta"])
          |> put_option(
            :compress_rotary_embedding_scaling_strategy,
            yarn_strategy(compress)
          )

        %{} = parameters ->
          put_option(opts, :rotary_embedding_base, parameters["rope_theta"])

        _other ->
          put_option(opts, :rotary_embedding_base, data["rope_theta"])
      end
    end

    defp yarn_strategy(%{"rope_type" => "yarn"} = parameters) do
      %{
        type: :yarn,
        factor: parameters["factor"] || 1.0,
        original_max_positions: parameters["original_max_position_embeddings"] || 4096,
        beta_fast: parameters["beta_fast"] || 32,
        beta_slow: parameters["beta_slow"] || 1
      }
    end

    defp yarn_strategy(_parameters), do: nil

    defp put_option(opts, _key, nil), do: opts
    defp put_option(opts, key, value), do: Keyword.put(opts, key, value)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.Layers.Moe

    import Bumblebee.Utils.Model, only: [join: 2]

    # DeepSeek V4 checkpoints use the naming of the original
    # implementation, rather than the huggingface/transformers module
    # names. Checkpoints saved by huggingface/transformers mix the two,
    # so the leaf names are looked up under both
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => ["model.embed_tokens", "model.embed"],
        "decoder.blocks.{n}.self_attention.query_down" => "model.layers.{n}.attn.wq_a",
        "decoder.blocks.{n}.self_attention.query_norm" => "model.layers.{n}.attn.q_norm",
        "decoder.blocks.{n}.self_attention.query_up" => "model.layers.{n}.attn.wq_b",
        "decoder.blocks.{n}.self_attention.key_value" => "model.layers.{n}.attn.wkv",
        "decoder.blocks.{n}.self_attention.key_value_norm" => [
          "model.layers.{n}.attn.norm",
          "model.layers.{n}.attn.kv_norm"
        ],
        "decoder.blocks.{n}.self_attention.output_up" => "model.layers.{n}.attn.wo_b",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.attn_norm",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.ffn_norm",
        "decoder.blocks.{n}.self_attention.compressor.key_value" =>
          "model.layers.{n}.attn.compressor.wkv",
        "decoder.blocks.{n}.self_attention.compressor.gate" =>
          "model.layers.{n}.attn.compressor.wgate",
        "decoder.blocks.{n}.self_attention.compressor.indexer.key_value" =>
          "model.layers.{n}.attn.indexer.compressor.wkv",
        "decoder.blocks.{n}.self_attention.compressor.indexer.gate" =>
          "model.layers.{n}.attn.indexer.compressor.wgate",
        "decoder.blocks.{n}.self_attention.compressor.indexer.query" =>
          "model.layers.{n}.attn.indexer.wq_b",
        "decoder.blocks.{n}.self_attention.compressor.indexer.head_weights" =>
          "model.layers.{n}.attn.indexer.weights_proj",
        "decoder.blocks.{n}.ffn.shared_expert.gate" => "model.layers.{n}.ffn.shared_experts.w1",
        "decoder.blocks.{n}.ffn.shared_expert.intermediate" =>
          "model.layers.{n}.ffn.shared_experts.w3",
        "decoder.blocks.{n}.ffn.shared_expert.output" => "model.layers.{n}.ffn.shared_experts.w2",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings,
            do: ["model.embed_tokens", "model.embed"],
            else: ["lm_head", "head"]
          ),
        "decoder.blocks.{n}.self_attention.sinks" => %{
          "sinks" => {[{"model.layers.{n}.attn", "attn_sink"}], fn [sinks] -> sinks end}
        },
        "decoder.blocks.{n}.self_attention.output_down" => %{
          "kernel" => {
            [{"model.layers.{n}.attn.wo_a", "weight"}],
            fn [kernel] ->
              {out_features, in_features} = Nx.shape(kernel)

              kernel
              |> Nx.reshape(
                {spec.output_groups, div(out_features, spec.output_groups), in_features}
              )
              |> Nx.transpose(axes: [0, 2, 1])
            end
          }
        }
      }
      |> Map.merge(hyper_connection_params())
      |> Map.merge(compressor_params())
      |> Map.merge(expert_params(spec))
    end

    defp hyper_connection_params() do
      %{
        "decoder.blocks.{n}.self_attention_hc" =>
          leaf_params("model.layers.{n}", "hc_attn", "model.layers.{n}.attn_hc"),
        "decoder.blocks.{n}.ffn_hc" =>
          leaf_params("model.layers.{n}", "hc_ffn", "model.layers.{n}.ffn_hc"),
        "output_hyper_connection" => leaf_params("model", "hc_head", "model.hc_head", "hc_")
      }
    end

    defp leaf_params(layer_name, prefix, module_name, module_prefix \\ "") do
      for {target, suffix} <- [{"kernel", "fn"}, {"base", "base"}, {"scale", "scale"}],
          into: %{} do
        {target,
         {
           [
             [
               {layer_name, prefix <> "_" <> suffix},
               {module_name, module_prefix <> suffix}
             ]
           ],
           fn [value] -> value end
         }}
      end
    end

    defp compressor_params() do
      for {target, source} <- [
            {"decoder.blocks.{n}.self_attention.compressor", "model.layers.{n}.attn.compressor"},
            {"decoder.blocks.{n}.self_attention.compressor.indexer",
             "model.layers.{n}.attn.indexer.compressor"}
          ],
          into: %{} do
        {join(target, "compress"),
         %{
           "position_bias" => {[{source, "ape"}], fn [bias] -> bias end},
           "weight" => {[{join(source, "norm"), "weight"}], fn [weight] -> weight end}
         }}
      end
    end

    defp expert_params(spec) do
      experts = "model.layers.{n}.ffn.experts"

      refs = fn projection ->
        for idx <- 0..(spec.num_experts - 1) do
          [
            {experts <> ".#{idx}.#{projection}", "weight"},
            {experts, packed_name(projection)}
          ]
        end
      end

      %{
        "decoder.blocks.{n}.ffn.router" => %{
          "kernel" => {
            [{"model.layers.{n}.ffn.gate", "weight"}],
            fn [kernel] -> Nx.transpose(kernel) end
          },
          "score_correction_bias" => {
            [
              [
                {"model.layers.{n}.ffn.gate", "bias"},
                {"model.layers.{n}.ffn.gate", "e_score_correction_bias"}
              ]
            ],
            fn [bias] -> bias end
          },
          "token_experts" => {
            [{"model.layers.{n}.ffn.gate", "tid2eid"}],
            fn [table] -> table end
          }
        },
        "decoder.blocks.{n}.ffn.experts" => %{
          "gate_kernel" =>
            {refs.("w1"), &Moe.stack_expert_kernels(&1, 0, spec.moe_intermediate_size)},
          "up_kernel" =>
            {refs.("w3"),
             &Moe.stack_expert_kernels(
               &1,
               spec.moe_intermediate_size,
               spec.moe_intermediate_size
             )},
          "down_kernel" => {refs.("w2"), &Moe.stack_expert_kernels(&1)}
        }
      }
    end

    defp packed_name("w2"), do: "down_proj"
    defp packed_name(_projection), do: "gate_up_proj"
  end
end
