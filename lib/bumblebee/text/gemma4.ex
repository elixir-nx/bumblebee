defmodule Bumblebee.Text.Gemma4 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 262_144,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 131_072,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 2560,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 10240,
        doc: "the dimensionality of intermediate layers"
      ],
      num_blocks: [
        default: 42,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 8,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 2,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      attention_head_size: [
        default: 256,
        doc: "the size of the key, value, and query projection per attention head for sliding attention layers"
      ],
      global_attention_head_size: [
        default: 512,
        doc: "the size of the key, value, and query projection per attention head for full attention layers"
      ],
      num_global_key_value_heads: [
        default: nil,
        doc: """
        the number of key value heads for full attention layers. When not set,
        defaults to `:num_key_value_heads`
        """
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 1_000_000.0,
        doc: "base for computing rotary embedding frequency for global (full) attention layers"
      ],
      rotary_embedding_base_local: [
        default: 10_000.0,
        doc: "base for computing rotary embedding frequency for local (sliding) attention layers"
      ],
      partial_rotary_factor: [
        default: 0.25,
        doc: "the fraction of dimensions to apply rotary embeddings to in full attention layers"
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
      attention_window_size: [
        default: 512,
        doc: "window size for the sliding attention window (used for `:sliding_attention` layers)"
      ],
      layer_types: [
        default: nil,
        doc: """
        a list of layer types for each layer, where each element is either `:sliding_attention`
        (local attention with sliding window) or `:full_attention` (global attention)
        """
      ],
      enable_moe_block: [
        default: false,
        doc: "whether to enable mixture-of-experts FFN blocks"
      ],
      num_experts: [
        default: nil,
        doc: "the number of experts in the mixture-of-experts block"
      ],
      top_k_experts: [
        default: nil,
        doc: "the number of top-k experts selected per token"
      ],
      moe_intermediate_size: [
        default: nil,
        doc: "the dimensionality of expert FFN intermediate layers"
      ],
      hidden_size_per_layer_input: [
        default: 0,
        doc: "the dimensionality of per-layer embeddings (PLE). Set to 0 to disable PLE"
      ],
      vocab_size_per_layer_input: [
        default: 262_144,
        doc: "the vocabulary size for per-layer embeddings"
      ],
      num_kv_shared_layers: [
        default: 0,
        doc: "the number of trailing layers that share KV from earlier layers of the same type"
      ],
      attention_k_eq_v: [
        default: false,
        doc: "whether to share key and value projections in full attention layers"
      ],
      final_logit_softcapping: [
        default: 30.0,
        doc: "the softcapping temperature for logits. Set to 0 to disable"
      ],
      tie_word_embeddings: [
        default: true,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 0)

  @moduledoc """
  Gemma 4 model family.

  Gemma 4 is an updated version of the Gemma architecture from Google DeepMind with
  several key improvements over Gemma 3:

    * Hybrid attention: alternating sliding window and full attention layers (5:1 ratio)
    * Dual RoPE: per-layer-type rotary embeddings (default for sliding, proportional for full)
    * Per-Layer Embeddings (PLE): per-layer token-dependent gating for efficiency
    * KV sharing: later layers reuse KV from earlier layers of the same attention type
    * Q/K/V normalization: RMS normalization on query, key, and value projections
    * Per-layer scalar: learned scaling factor per transformer block
    * Optional MoE: mixture-of-experts FFN blocks (26B-A4B variant)

  This module implements the text-only portion of Gemma 4. The multimodal
  `Gemma4ForConditionalGeneration` model is not yet supported.

  Note: this model uses a custom decoder loop rather than `Layers.Transformer.blocks/2`
  because it requires features not currently available in the shared infrastructure:
  per-layer embeddings (PLE) threaded through the block loop, cross-block KV sharing
  state, per-layer head dimension variation, per-layer scalar, and value normalization.

  ## Architectures

    * `:base` - plain Gemma 4 text model without any head on top

    * `:for_causal_language_modeling` - Gemma 4 with a language modeling
      head. The head returns logits for each token in the original
      sequence

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

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling
    ]

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

  # Custom init_cache is needed because Gemma 4 uses different head dimensions
  # per layer type (attention_head_size for sliding, global_attention_head_size
  # for full attention), so the standard Layers.Decoder.init_cache (which creates
  # uniform cache shapes) cannot be used directly.
  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    layer_types = resolve_layer_types(spec)

    blocks_cache =
      layer_types
      |> Enum.map(fn layer_type ->
        head_size =
          case layer_type do
            :full_attention -> spec.global_attention_head_size || spec.attention_head_size
            _ -> spec.attention_head_size
          end

        kv_heads =
          if layer_type == :full_attention && spec.num_global_key_value_heads do
            spec.num_global_key_value_heads
          else
            spec.num_key_value_heads
          end

        shape = {batch_size, max_length, kv_heads, head_size}
        zeros = Nx.broadcast(Nx.tensor(0.0), shape)

        %{self_attention: %{key: zeros, value: zeros}, cross_attention: %Axon.None{}}
      end)
      |> List.to_tuple()

    offset = Nx.tensor(0)
    attention_mask = Nx.broadcast(0, {batch_size, max_length})

    %{blocks: blocks_cache, offset: offset, attention_mask: attention_mask}
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

    logits =
      if spec.final_logit_softcapping && spec.final_logit_softcapping > 0 do
        cap = spec.final_logit_softcapping

        Axon.nx(logits, fn x ->
          x
          |> Nx.divide(cap)
          |> Nx.tanh()
          |> Nx.multiply(cap)
        end)
      else
        logits
      end

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

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(embeddings)
      end

    # PLE: compute per-layer inputs from a separate embedding
    per_layer_inputs =
      if spec.hidden_size_per_layer_input > 0 do
        ple_embeddings(inputs["input_ids"], embeddings, spec, name: "ple")
      else
        Layers.none()
      end

    # Decode through all blocks (custom loop, not Layers.Transformer.blocks)
    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        attention_mask,
        inputs["cache"],
        per_layer_inputs,
        spec,
        name: "decoder"
      )

    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    %{
      hidden_state: hidden_state,
      hidden_states: decoder_outputs.hidden_states,
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    Layers.default input_embeddings do
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
    |> Axon.nx(fn x ->
      scale =
        spec.hidden_size
        |> Nx.tensor(type: Nx.type(x))
        |> Nx.sqrt()

      Nx.multiply(x, scale)
    end)
  end

  defp ple_embeddings(input_ids, main_embeddings, spec, opts) do
    name = opts[:name]
    n = spec.num_blocks
    d = spec.hidden_size_per_layer_input

    # Separate embedding: vocab → (num_blocks * ple_dim)
    raw_ple =
      Axon.embedding(input_ids, spec.vocab_size_per_layer_input, n * d,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "embed_tokens_per_layer")
      )

    # Scale by sqrt(ple_dim)
    raw_ple =
      Axon.nx(raw_ple, fn x ->
        scale = d |> Nx.tensor(type: Nx.type(x)) |> Nx.sqrt()
        Nx.multiply(x, scale)
      end)

    # Project main embeddings → per-layer space
    projection =
      Axon.dense(main_embeddings, n * d,
        use_bias: false,
        name: join(name, "per_layer_model_projection")
      )

    projection =
      Axon.nx(projection, fn x ->
        scale = spec.hidden_size |> Nx.tensor(type: Nx.type(x)) |> Nx.rsqrt()
        Nx.multiply(x, scale)
      end)

    # Reshape to {batch, seq, num_blocks, ple_dim}
    projection =
      Axon.nx(projection, fn x ->
        {batch, seq, _flat} = Nx.shape(x)
        Nx.reshape(x, {batch, seq, n, d})
      end)

    # Normalize each layer's projection (Gemma4RMSNorm: weight*normed)
    projection =
      Axon.layer(
        fn x, scale, _opts ->
          {batch, seq, num_layers, dim} = Nx.shape(x)
          flat = Nx.reshape(x, {batch * seq * num_layers, dim})
          normed = rms_norm_no_weight(flat, spec.layer_norm_epsilon)
          normed = Nx.multiply(normed, scale)
          Nx.reshape(normed, {batch, seq, num_layers, dim})
        end,
        [projection, Axon.param("weight", fn _shape -> {d} end, initializer: :zeros)],
        name: join(name, "per_layer_projection_norm")
      )

    # Reshape raw PLE and combine: (projection + ple) * 2^(-0.5)
    Axon.layer(
      fn proj, ple, _opts ->
        {batch, seq, _flat} = Nx.shape(ple)
        ple_reshaped = Nx.reshape(ple, {batch, seq, n, d})
        scale = Nx.tensor(:math.pow(2.0, -0.5), type: Nx.type(proj))
        Nx.multiply(Nx.add(proj, ple_reshaped), scale)
      end,
      [projection, raw_ple],
      name: join(name, "combine")
    )
  end

  defp decoder(hidden_state, position_ids, attention_mask, _cache, per_layer_inputs, spec, opts) do
    name = opts[:name]
    layer_types = resolve_layer_types(spec)

    # KV sharing: compute which layers share KV and with whom
    first_kv_shared = spec.num_blocks - spec.num_kv_shared_layers
    non_shared_types = Enum.take(layer_types, first_kv_shared)

    kv_share_map =
      for {_lt, idx} <- Enum.with_index(layer_types), idx >= first_kv_shared, into: %{} do
        this_type = Enum.at(layer_types, idx)
        # Find last non-shared layer of the same type
        source_idx =
          non_shared_types
          |> Enum.with_index()
          |> Enum.filter(fn {t, _i} -> t == this_type end)
          |> List.last()
          |> elem(1)

        {idx, source_idx}
      end

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.none(),
      attentions: Layers.none(),
      cache: Layers.none(),
      stored_kv: %{}
    }

    Enum.reduce(Enum.with_index(layer_types), state, fn {layer_type, idx}, acc ->
      block_name = join(name, "blocks.#{idx}")

      # Extract per-layer input for PLE
      ple_input =
        if spec.hidden_size_per_layer_input > 0 do
          Axon.nx(per_layer_inputs, fn x ->
            x[[.., .., idx, ..]]
          end)
        else
          Layers.none()
        end

      # KV sharing: determine if this layer uses shared KV
      shared_kv_source = Map.get(kv_share_map, idx)
      shared_kv = if shared_kv_source, do: Map.get(acc.stored_kv, shared_kv_source), else: nil

      {new_hidden, kv_out} =
        decoder_block(
          acc.hidden_state,
          position_ids,
          attention_mask,
          ple_input,
          layer_type,
          idx,
          spec,
          shared_kv: shared_kv,
          name: block_name
        )

      # Store KV from non-shared layers
      new_stored_kv =
        if idx < first_kv_shared and kv_out do
          Map.put(acc.stored_kv, idx, kv_out)
        else
          acc.stored_kv
        end

      %{
        acc
        | hidden_state: new_hidden,
          hidden_states: Layers.append(acc.hidden_states, new_hidden),
          stored_kv: new_stored_kv
      }
    end)
  end

  defp decoder_block(
         hidden_state,
         position_ids,
         attention_mask,
         ple_input,
         layer_type,
         layer_idx,
         spec,
         opts
       ) do
    name = opts[:name]
    shared_kv = opts[:shared_kv]

    hd_dim = if layer_type == :full_attention, do: spec.global_attention_head_size || spec.attention_head_size, else: spec.attention_head_size

    kv_heads =
      if layer_type == :full_attention && spec.attention_k_eq_v && spec.num_global_key_value_heads do
        spec.num_global_key_value_heads
      else
        spec.num_key_value_heads
      end

    share_kv_proj = layer_type == :full_attention && spec.attention_k_eq_v

    normed = rms_norm_llama(hidden_state, spec, name: join(name, "input_layernorm"))

    {attn_out, kv_out} =
      self_attention(
        normed,
        position_ids,
        attention_mask,
        layer_type,
        layer_idx,
        spec,
        hd_dim: hd_dim,
        kv_heads: kv_heads,
        share_kv: share_kv_proj,
        shared_kv: shared_kv,
        name: join(name, "self_attn")
      )

    # Post-attention norm
    attn_out = rms_norm_llama(attn_out, spec, name: join(name, "post_attention_layernorm"))
    hidden_state = Axon.add(hidden_state, attn_out)

    # Pre-FFN norm → MLP
    residual = hidden_state
    normed_for_ffn = rms_norm_llama(hidden_state, spec, name: join(name, "pre_feedforward_layernorm"))
    mlp_out = gated_ffn(normed_for_ffn, spec.intermediate_size, spec.hidden_size, spec, name: join(name, "mlp"))

    # Optional MoE
    hidden_state =
      if spec.enable_moe_block do
        mlp_normed = rms_norm_llama(mlp_out, spec, name: join(name, "post_feedforward_layernorm_1"))

        moe_input = rms_norm_llama(residual, spec, name: join(name, "pre_feedforward_layernorm_2"))
        moe_out = moe_block(moe_input, spec, name: join(name, "moe"))
        moe_normed = rms_norm_llama(moe_out, spec, name: join(name, "post_feedforward_layernorm_2"))

        Axon.add(mlp_normed, moe_normed)
      else
        mlp_out
      end

    # Post-FFN norm
    hidden_state = rms_norm_llama(hidden_state, spec, name: join(name, "post_feedforward_layernorm"))
    hidden_state = Axon.add(residual, hidden_state)

    # PLE gating
    hidden_state =
      if spec.hidden_size_per_layer_input > 0 do
        ple_gate(hidden_state, ple_input, spec, name: join(name, "ple"))
      else
        hidden_state
      end

    # Per-layer scalar
    hidden_state =
      Axon.layer(
        fn x, scalar, _opts -> Nx.multiply(x, Nx.squeeze(scalar)) end,
        [hidden_state, Axon.param("layer_scalar", fn _shape -> {1} end, initializer: fn _, _ -> Nx.tensor([1.0]) end)],
        name: join(name, "layer_scalar")
      )

    {hidden_state, kv_out}
  end

  defp self_attention(
         hidden_state,
         position_ids,
         attention_mask,
         layer_type,
         _layer_idx,
         spec,
         opts
       ) do
    name = opts[:name]
    hd_dim = opts[:hd_dim]
    kv_heads = opts[:kv_heads]
    share_kv = opts[:share_kv]
    shared_kv = opts[:shared_kv]

    q_size = spec.num_attention_heads * hd_dim
    kv_size = kv_heads * hd_dim

    # Query projection
    query = Axon.dense(hidden_state, q_size, use_bias: false, name: join(name, "q_proj"))

    query = Axon.nx(query, fn q ->
      {b, s, _} = Nx.shape(q)
      Nx.reshape(q, {b, s, spec.num_attention_heads, hd_dim})
    end)

    query = rms_norm_per_head(query, hd_dim, spec, name: join(name, "q_norm"))

    # K/V projections (compute fresh or reuse from shared layer)
    {key, value, kv_out} =
      if shared_kv do
        {shared_kv.key, shared_kv.value, nil}
      else
        key = Axon.dense(hidden_state, kv_size, use_bias: false, name: join(name, "k_proj"))
        value =
          if share_kv, do: key,
          else: Axon.dense(hidden_state, kv_size, use_bias: false, name: join(name, "v_proj"))

        key = Axon.nx(key, fn k ->
          {b, s, _} = Nx.shape(k)
          Nx.reshape(k, {b, s, kv_heads, hd_dim})
        end)

        value = Axon.nx(value, fn v ->
          {b, s, _} = Nx.shape(v)
          Nx.reshape(v, {b, s, kv_heads, hd_dim})
        end)

        key = rms_norm_per_head(key, hd_dim, spec, name: join(name, "k_norm"))

        value =
          Axon.layer(
            fn v, _opts -> rms_norm_no_weight(v, spec.layer_norm_epsilon) end,
            [value],
            name: join(name, "v_norm")
          )

        # RoPE on K
        {rope_base, partial_factor} =
          case layer_type do
            :full_attention -> {spec.rotary_embedding_base, spec.partial_rotary_factor}
            _ -> {spec.rotary_embedding_base_local, 1.0}
          end

        {_, key_roped} =
          apply_rope(key, key, position_ids, hd_dim, rope_base, partial_factor, name: join(name, "rope_k"))

        key_t = Axon.nx(key_roped, &Nx.transpose(&1, axes: [0, 2, 1, 3]))
        value_t = Axon.nx(value, &Nx.transpose(&1, axes: [0, 2, 1, 3]))

        {key_t, value_t, %{key: key_t, value: value_t}}
      end

    # RoPE on Q
    {rope_base, partial_factor} =
      case layer_type do
        :full_attention -> {spec.rotary_embedding_base, spec.partial_rotary_factor}
        _ -> {spec.rotary_embedding_base_local, 1.0}
      end

    {query_roped, _} =
      apply_rope(query, query, position_ids, hd_dim, rope_base, partial_factor, name: join(name, "rope_q"))

    query = Axon.nx(query_roped, &Nx.transpose(&1, axes: [0, 2, 1, 3]))

    # GQA: repeat KV heads to match query heads (repeat_interleave semantics)
    num_groups = div(spec.num_attention_heads, kv_heads)

    key =
      if num_groups > 1 do
        Axon.nx(key, fn k ->
          {b, h, s, d} = Nx.shape(k)

          k
          |> Nx.reshape({b, h, 1, s, d})
          |> Nx.broadcast({b, h, num_groups, s, d})
          |> Nx.reshape({b, h * num_groups, s, d})
        end)
      else
        key
      end

    value =
      if num_groups > 1 do
        Axon.nx(value, fn v ->
          {b, h, s, d} = Nx.shape(v)

          v
          |> Nx.reshape({b, h, 1, s, d})
          |> Nx.broadcast({b, h, num_groups, s, d})
          |> Nx.reshape({b, h * num_groups, s, d})
        end)
      else
        value
      end

    # Scaled dot-product attention
    attn_output =
      Axon.layer(
        fn q, k, v, mask, _opts ->
          # Attention scaling is 1.0 since Q/K are already RMS-normalized
          scores = q |> Nx.dot([3], [0, 1], k, [3], [0, 1])

          # Causal mask
          {_b, _h, q_len, kv_len} = Nx.shape(scores)

          causal =
            Nx.iota({q_len, kv_len}, axis: 1)
            |> Nx.less_equal(Nx.iota({q_len, kv_len}, axis: 0) |> Nx.add(kv_len - q_len))

          # Sliding window mask
          causal =
            case layer_type do
              :sliding_attention ->
                window = spec.attention_window_size
                row_idx = Nx.iota({q_len, kv_len}, axis: 0) |> Nx.add(kv_len - q_len)
                col_idx = Nx.iota({q_len, kv_len}, axis: 1)
                dist = Nx.subtract(row_idx, col_idx)
                Nx.logical_and(causal, Nx.less_equal(dist, window))

              _ ->
                causal
            end

          # Broadcast to match scores shape
          causal = Nx.reshape(causal, {1, 1, q_len, kv_len})
          causal = Nx.broadcast(causal, Nx.shape(scores))
          mask_value = Nx.Constants.neg_infinity(Nx.type(scores))
          scores = Nx.select(causal, scores, mask_value)

          # Apply padding mask if present
          scores =
            case mask do
              %Axon.None{} -> scores
              m ->
                pad_mask = Nx.reshape(m, {Nx.axis_size(m, 0), 1, 1, Nx.axis_size(m, 1)})
                pad_mask = Nx.broadcast(pad_mask, Nx.shape(scores))
                Nx.select(pad_mask, scores, mask_value)
            end

          weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
          weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

          Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
        end,
        [query, key, value, attention_mask],
        name: join(name, "attention")
      )

    # Reshape back: {batch, seq, num_heads * head_dim}
    attn_output =
      Axon.nx(attn_output, fn x ->
        {b, h, s, d} = Nx.shape(x)
        x |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({b, s, h * d})
      end)

    # Output projection
    out = Axon.dense(attn_output, spec.hidden_size, use_bias: false, name: join(name, "o_proj"))
    {out, kv_out}
  end


  defp apply_rope(query, key, position_ids, head_dim, theta, partial_factor, opts) do
    name = opts[:name]
    half_dim = div(head_dim, 2)
    rope_angles = round(head_dim * partial_factor / 2)

    applied =
      Axon.layer(
        fn q, k, pos, _opts ->
          # Compute inv_freq (proportional RoPE: frequency scale uses full head_dim)
          inv_freq_rotated =
            Nx.iota({rope_angles})
            |> Nx.multiply(2)
            |> Nx.divide(head_dim)
            |> Nx.negate()
            |> Nx.multiply(Nx.log(Nx.tensor(theta, type: :f32)))
            |> Nx.exp()
            |> Nx.as_type(Nx.type(q))

          # Zero-pad for non-rotated dims (matches Python proportional RoPE)
          nope_angles = half_dim - rope_angles

          inv_freq =
            if nope_angles > 0 do
              zeros = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {nope_angles})
              Nx.concatenate([inv_freq_rotated, zeros])
            else
              inv_freq_rotated
            end

          pos_f = Nx.as_type(pos, Nx.type(q))
          pos_expanded = Nx.new_axis(pos_f, -1)

          # freqs: {batch, seq, half_dim}
          freqs = Nx.multiply(pos_expanded, Nx.reshape(inv_freq, {1, 1, half_dim}))

          # emb = cat(freqs, freqs) → {batch, seq, head_dim}
          emb = Nx.concatenate([freqs, freqs], axis: -1)
          cos = Nx.cos(emb) |> Nx.new_axis(2)
          sin = Nx.sin(emb) |> Nx.new_axis(2)

          q_rotated = rotate_half(q, cos, sin)
          k_rotated = rotate_half(k, cos, sin)

          {q_rotated, k_rotated}
        end,
        [query, key, position_ids],
        name: name
      )

    q_out = Axon.nx(applied, &elem(&1, 0))
    k_out = Axon.nx(applied, &elem(&1, 1))
    {q_out, k_out}
  end

  defp rotate_half(x, cos, sin) do
    {_b, _s, _h, d} = Nx.shape(x)
    half = div(d, 2)
    x1 = x[[.., .., .., 0..(half - 1)]]
    x2 = x[[.., .., .., half..(d - 1)]]
    rotated = Nx.concatenate([Nx.negate(x2), x1], axis: -1)
    Nx.add(Nx.multiply(x, cos), Nx.multiply(rotated, sin))
  end


  defp gated_ffn(hidden_state, intermediate_size, output_size, _spec, opts) do
    name = opts[:name]

    gate = Axon.dense(hidden_state, intermediate_size, use_bias: false, name: join(name, "gate_proj"))
    up = Axon.dense(hidden_state, intermediate_size, use_bias: false, name: join(name, "up_proj"))

    hidden =
      Axon.layer(
        fn g, u, _opts -> Nx.multiply(Bumblebee.Layers.gelu_approx_tanh(g), u) end,
        [gate, up],
        name: join(name, "gate_mul")
      )

    Axon.dense(hidden, output_size, use_bias: false, name: join(name, "down_proj"))
  end


  defp moe_block(hidden_state, spec, opts) do
    name = opts[:name]

    # Flatten to 2D for routing
    flat = Axon.nx(hidden_state, fn x ->
      {b, s, d} = Nx.shape(x)
      Nx.reshape(x, {b * s, d})
    end)

    # Combined router + experts in a single Axon.layer to avoid map outputs
    expert_out =
      Axon.layer(
        fn x, proj_w, scale_w, expert_scale_w, gate_up_w, down_w, _opts ->
          # Router
          normed = rms_norm_no_weight(x, spec.layer_norm_epsilon)
          scalar_root = :math.pow(spec.hidden_size, -0.5)
          scaled = Nx.multiply(Nx.multiply(normed, scale_w), scalar_root)

          scores = Nx.dot(scaled, Nx.transpose(proj_w))
          probs = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
          probs = Nx.divide(probs, Nx.sum(probs, axes: [-1], keep_axes: true))

          {top_weights, top_indices} = Nx.top_k(probs, k: spec.top_k_experts)
          top_weights = Nx.divide(top_weights, Nx.sum(top_weights, axes: [-1], keep_axes: true))
          expert_scales = Nx.take(expert_scale_w, top_indices)
          top_weights = Nx.multiply(top_weights, expert_scales)

          # Expert computation
          {n_tokens, _hidden} = Nx.shape(x)
          k = spec.top_k_experts
          mid = spec.moe_intermediate_size

          Enum.reduce(0..(k - 1), Nx.broadcast(Nx.tensor(0.0, type: Nx.type(x)), Nx.shape(x)), fn ki, acc ->
            expert_idx = top_indices[[.., ki]]
            weight = top_weights[[.., ki]] |> Nx.reshape({n_tokens, 1})

            gate_up = Nx.take(gate_up_w, expert_idx)
            down = Nx.take(down_w, expert_idx)

            gate_proj = gate_up[[.., 0..(mid - 1), ..]]
            up_proj = gate_up[[.., mid..(2 * mid - 1), ..]]

            gate_out = Nx.dot(Nx.new_axis(x, 1), [2], [0], gate_proj, [2], [0]) |> Nx.squeeze(axes: [1])
            up_out = Nx.dot(Nx.new_axis(x, 1), [2], [0], up_proj, [2], [0]) |> Nx.squeeze(axes: [1])
            activated = Nx.multiply(Bumblebee.Layers.gelu_approx_tanh(gate_out), up_out)
            expert_output = Nx.dot(Nx.new_axis(activated, 1), [2], [0], down, [2], [0]) |> Nx.squeeze(axes: [1])

            Nx.add(acc, Nx.multiply(expert_output, weight))
          end)
        end,
        [
          flat,
          Axon.param("router.proj.weight", fn _shape -> {spec.num_experts, spec.hidden_size} end),
          Axon.param("router.scale", fn _shape -> {spec.hidden_size} end, initializer: fn _, _ -> Nx.broadcast(1.0, {spec.hidden_size}) end),
          Axon.param("router.per_expert_scale", fn _shape -> {spec.num_experts} end, initializer: fn _, _ -> Nx.broadcast(1.0, {spec.num_experts}) end),
          Axon.param("experts.gate_up_proj", fn _shape -> {spec.num_experts, 2 * spec.moe_intermediate_size, spec.hidden_size} end),
          Axon.param("experts.down_proj", fn _shape -> {spec.num_experts, spec.hidden_size, spec.moe_intermediate_size} end)
        ],
        name: name
      )

    # Reshape back to {batch, seq, hidden}
    Axon.layer(
      fn flat_out, original, _opts ->
        Nx.reshape(flat_out, Nx.shape(original))
      end,
      [expert_out, hidden_state],
      name: join(name, "reshape")
    )
  end


  defp ple_gate(hidden_state, ple_input, spec, opts) do
    name = opts[:name]
    d = spec.hidden_size_per_layer_input

    gated =
      Axon.dense(hidden_state, d, use_bias: false, name: join(name, "per_layer_input_gate"))

    gated = Layers.activation(gated, :gelu_approx_tanh)

    multiplied =
      Axon.layer(
        fn g, ple, _opts -> Nx.multiply(g, ple) end,
        [gated, ple_input],
        name: join(name, "multiply")
      )

    projected =
      Axon.dense(multiplied, spec.hidden_size, use_bias: false, name: join(name, "per_layer_projection"))

    # post_per_layer_input_norm: Gemma4RMSNorm (weight * normed, no shift)
    normed =
      Layers.rms_norm(projected,
        name: join(name, "post_per_layer_input_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    Axon.add(hidden_state, normed)
  end


  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    if spec.tie_word_embeddings do
      Layers.dense_transposed(hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output")
      )
    else
      Axon.dense(hidden_state, spec.vocab_size,
        use_bias: false,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output")
      )
    end
  end


  defp rms_norm_llama(input, spec, opts) do
    Layers.rms_norm(input,
      name: opts[:name],
      epsilon: spec.layer_norm_epsilon,
      upcast: :all
    )
  end

  defp rms_norm_per_head(x, head_dim, spec, opts) do
    Axon.layer(
      fn tensor, scale, _opts ->
        variance = Nx.mean(Nx.multiply(tensor, tensor), axes: [-1], keep_axes: true)
        normed = Nx.multiply(tensor, Nx.rsqrt(Nx.add(variance, spec.layer_norm_epsilon)))
        scale_broadcast = Nx.reshape(scale, {1, 1, 1, head_dim})
        Nx.multiply(normed, scale_broadcast)
      end,
      [x, Axon.param("weight", fn _shape -> {head_dim} end, initializer: :zeros)],
      name: opts[:name]
    )
  end

  defp rms_norm_no_weight(x, epsilon) do
    variance = Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true)
    Nx.multiply(x, Nx.rsqrt(Nx.add(variance, epsilon)))
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp resolve_layer_types(spec) do
    if spec.layer_types do
      spec.layer_types
    else
      # Default pattern: 5 sliding + 1 full
      for i <- 0..(spec.num_blocks - 1) do
        if rem(i + 1, 6) == 0, do: :full_attention, else: :sliding_attention
      end
    end
  end


  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      # Gemma 4 nests text config under "text_config"
      text = Map.get(data, "text_config", data)

      layer_types_converter = fn _name, value ->
        types =
          Enum.map(value, fn
            "sliding_attention" -> :sliding_attention
            "full_attention" -> :full_attention
            other -> String.to_atom(other)
          end)

        {:ok, types}
      end

      rope_params = Map.get(text, "rope_parameters", %{})
      sliding_rope = Map.get(rope_params, "sliding_attention", %{})
      full_rope = Map.get(rope_params, "full_attention", %{})

      opts =
        convert!(text,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          global_attention_head_size: {"global_head_dim", optional(number())},
          num_global_key_value_heads: {"num_global_key_value_heads", optional(number())},
          activation: {"hidden_activation", activation()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          attention_window_size: {"sliding_window", number()},
          layer_types: {"layer_types", layer_types_converter},
          enable_moe_block: {"enable_moe_block", boolean()},
          num_experts: {"num_experts", optional(number())},
          top_k_experts: {"top_k_experts", optional(number())},
          moe_intermediate_size: {"moe_intermediate_size", optional(number())},
          hidden_size_per_layer_input: {"hidden_size_per_layer_input", number()},
          vocab_size_per_layer_input: {"vocab_size_per_layer_input", optional(number())},
          num_kv_shared_layers: {"num_kv_shared_layers", number()},
          attention_k_eq_v: {"attention_k_eq_v", boolean()},
          final_logit_softcapping: {"final_logit_softcapping", optional(number())},
          tie_word_embeddings: {"tie_word_embeddings", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      opts =
        opts
        |> Keyword.put(:rotary_embedding_base_local, Map.get(sliding_rope, "rope_theta", 10_000.0))
        |> Keyword.put(:rotary_embedding_base, Map.get(full_rope, "rope_theta", 1_000_000.0))
        |> Keyword.put(:partial_rotary_factor, Map.get(full_rope, "partial_rotary_factor", 0.25))

      @for.config(spec, opts)
    end
  end


  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "ple.embed_tokens_per_layer" => "model.embed_tokens_per_layer",
        "ple.per_layer_model_projection" => "model.per_layer_model_projection",
        "ple.per_layer_projection_norm" => "model.per_layer_projection_norm",
        "decoder.blocks.{n}.input_layernorm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.self_attn.q_proj" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attn.k_proj" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attn.v_proj" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attn.o_proj" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attn.q_norm" => "model.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attn.k_norm" => "model.layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.post_attention_layernorm" =>
          "model.layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.pre_feedforward_layernorm" =>
          "model.layers.{n}.pre_feedforward_layernorm",
        "decoder.blocks.{n}.mlp.gate_proj" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.mlp.up_proj" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.mlp.down_proj" => "model.layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.post_feedforward_layernorm" =>
          "model.layers.{n}.post_feedforward_layernorm",
        "decoder.blocks.{n}.post_feedforward_layernorm_1" =>
          "model.layers.{n}.post_feedforward_layernorm_1",
        "decoder.blocks.{n}.pre_feedforward_layernorm_2" =>
          "model.layers.{n}.pre_feedforward_layernorm_2",
        "decoder.blocks.{n}.post_feedforward_layernorm_2" =>
          "model.layers.{n}.post_feedforward_layernorm_2",
        "decoder.blocks.{n}.moe" => %{
          "router.proj.weight" => {
            [{"model.layers.{n}.router.proj", "weight"}],
            fn [value] -> value end
          },
          "router.scale" => {
            [{"model.layers.{n}.router", "scale"}],
            fn [value] -> value end
          },
          "router.per_expert_scale" => {
            [{"model.layers.{n}.router", "per_expert_scale"}],
            fn [value] -> value end
          },
          "experts.gate_up_proj" => {
            [{"model.layers.{n}.experts", "gate_up_proj"}],
            fn [value] -> value end
          },
          "experts.down_proj" => {
            [{"model.layers.{n}.experts", "down_proj"}],
            fn [value] -> value end
          }
        },
        "decoder.blocks.{n}.ple.per_layer_input_gate" =>
          "model.layers.{n}.per_layer_input_gate",
        "decoder.blocks.{n}.ple.per_layer_projection" =>
          "model.layers.{n}.per_layer_projection",
        "decoder.blocks.{n}.ple.post_per_layer_input_norm" =>
          "model.layers.{n}.post_per_layer_input_norm",
        "decoder.blocks.{n}.layer_scalar" => "model.layers.{n}",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings,
            do: "model.embed_tokens",
            else: "lm_head"
          )
      }
    end
  end
end
