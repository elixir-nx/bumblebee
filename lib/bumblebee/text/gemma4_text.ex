defmodule Bumblebee.Text.Gemma4Text do
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
        the maximum sequence length that this model can process
        """
      ],
      hidden_size: [
        default: 2304,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 9216,
        doc: "the dimensionality of intermediate layers"
      ],
      attention_head_size: [
        default: 256,
        doc:
          "the size of the key, value, and query projection per attention head for sliding attention layers"
      ],
      global_attention_head_size: [
        default: 512,
        doc:
          "the size of the key, value, and query projection per attention head for global (full) attention layers"
      ],
      num_blocks: [
        default: 30,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 8,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 4,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      num_global_key_value_heads: [
        default: nil,
        doc: """
        the number of key value heads for global (full) attention layers.
        If nil, defaults to num_key_value_heads.
        """
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 1_000_000,
        doc: "base for computing rotary embedding frequency for global attention layers"
      ],
      rotary_embedding_base_local: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency for local (sliding) attention layers"
      ],
      partial_rotary_factor: [
        default: 1.0,
        doc: """
        the fraction of head dimensions to apply rotary embeddings to in global attention layers.
        Sliding attention layers always use full rotation (1.0).
        Extracted from rope_parameters.full_attention.partial_rotary_factor.
        """
      ],
      use_attention_bias: [
        default: false,
        doc:
          "whether or not to use bias in the query, key, value, and output projections in attention layers"
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
        doc:
          "window size for both sides of the sliding attention window (used for `:sliding_attention` layers)"
      ],
      layer_types: [
        default: nil,
        doc: """
        a list of layer types for each layer, where each element is either `:sliding_attention`
        (local attention with sliding window) or `:full_attention` (global attention)
        """
      ],
      tie_word_embeddings: [
        default: true,
        doc: "whether to tie input and output embedding weights"
      ],
      final_logit_softcapping: [
        default: nil,
        doc: """
        if set, logits are capped using `tanh` to this value before the final softmax.
        This prevents extreme logit values from dominating the output distribution.
        Logits are scaled by tanh(logit / cap) * cap.
        """
      ],
      hidden_size_per_layer_input: [
        default: 256,
        doc: """
        the dimensionality of the per-layer input embeddings (PLE). Each transformer layer
        gets its own small embedding that is added to the main hidden state.
        """
      ],
      vocab_size_per_layer_input: [
        default: 262_144,
        doc: "the vocabulary size for per-layer input embeddings"
      ],
      num_kv_shared_layers: [
        default: 0,
        doc: """
        the number of consecutive decoder layers that share the same key-value projections.
        A value of 0 means no sharing (each layer has independent KV projections).
        """
      ],
      use_double_wide_mlp: [
        default: false,
        doc: """
        whether to use a double-width MLP with fused gate and up projections.
        When true, the gate and up projections are doubled in size.
        """
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++ Shared.token_options(pad_token_id: 0)

  @moduledoc """
  Gemma 4 model family (text backbone).

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
      :for_causal_language_modeling,
      :for_sequence_classification
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

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    layer_types = spec.layer_types || generate_layer_types(spec.num_blocks)

    blocks =
      Enum.map(0..(spec.num_blocks - 1), fn idx ->
        head_size =
          case Enum.at(layer_types, idx, :sliding_attention) do
            :full_attention -> spec.global_attention_head_size
            :sliding_attention -> spec.attention_head_size
          end

        shape = {batch_size, max_length, spec.num_attention_heads, head_size}
        zeros = Nx.broadcast(0.0, shape)
        self_attention = %{key: zeros, value: zeros}

        %{self_attention: self_attention, cross_attention: %Axon.None{}}
      end)
      |> List.to_tuple()

    offset = Nx.tensor(0)
    attention_mask = Nx.broadcast(0, {batch_size, max_length})
    %{blocks: blocks, offset: offset, attention_mask: attention_mask}
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

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output",
        use_bias: false
      )

    pooled_logits =
      Layers.if_present inputs["input_ids"] do
        Axon.layer(
          fn logits, input_ids, _opts ->
            indices =
              input_ids
              |> Nx.not_equal(spec.pad_token_id)
              |> Nx.sum(axes: [-1])
              |> Nx.subtract(1)
              |> Nx.as_type({:s, 64})

            Bumblebee.Utils.Nx.batched_take(logits, indices)
          end,
          [logits, inputs["input_ids"]]
        )
      else
        Layers.take_token(logits, axis: 1, index: -1)
      end

    Layers.output(%{
      logits: pooled_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(
        inputs["input_ids"],
        inputs["input_embeddings"],
        spec,
        name: "embedder"
      )

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    # PLE: compute per-layer inputs
    per_layer_inputs =
      if spec.hidden_size_per_layer_input do
        compute_per_layer_inputs(inputs["input_ids"], embeddings, spec)
      else
        nil
      end

    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        per_layer_inputs: per_layer_inputs,
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
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp compute_per_layer_inputs(input_ids, embeddings, spec) do
    ple_dim = spec.hidden_size_per_layer_input
    num_layers = spec.num_blocks
    total_ple_dim = num_layers * ple_dim

    # Token-identity: lookup in per-layer embedding table
    token_identity =
      Axon.embedding(input_ids, spec.vocab_size_per_layer_input, total_ple_dim,
        name: "embedder.token_embedding_per_layer"
      )
      |> Axon.nx(fn x ->
        # Scale by sqrt(ple_dim)
        scale = Nx.tensor(ple_dim, type: Nx.type(x)) |> Nx.sqrt()
        x = Nx.multiply(x, scale)
        # Reshape from [B, S, num_layers * ple_dim] to [B, S, num_layers, ple_dim]
        shape = Nx.shape(x)
        batch = elem(shape, 0)
        seq = elem(shape, 1)
        Nx.reshape(x, {batch, seq, num_layers, ple_dim})
      end)

    # Context-aware: project main embeddings, reshape, then norm.
    # The norm weight has shape [ple_dim] because HuggingFace applies it
    # after reshaping to [B, S, num_layers, ple_dim].
    context_aware =
      Axon.dense(embeddings, total_ple_dim,
        name: "per_layer_model_projection",
        use_bias: false
      )
      |> Axon.nx(fn x ->
        scale = Nx.divide(1.0, Nx.sqrt(Nx.tensor(spec.hidden_size, type: Nx.type(x))))
        x = Nx.multiply(x, scale)
        shape = Nx.shape(x)
        batch = elem(shape, 0)
        seq = elem(shape, 1)
        Nx.reshape(x, {batch, seq, num_layers, ple_dim})
      end)
      |> Layers.rms_norm(
        name: "per_layer_projection_norm",
        epsilon: spec.layer_norm_epsilon
      )

    # Combine: (token_identity + context_aware) * (1/sqrt(2))
    Axon.layer(
      fn token_id, context, _opts ->
        inv_sqrt2 = Nx.tensor(1.0 / :math.sqrt(2), type: Nx.type(token_id))
        Nx.multiply(Nx.add(token_id, context), inv_sqrt2)
      end,
      [token_identity, context_aware]
    )
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
      normalization_factor =
        spec.hidden_size
        |> Nx.tensor(type: Nx.type(x))
        |> Nx.sqrt()

      Nx.multiply(x, normalization_factor)
    end)
  end

  defp decoder(
         hidden_state,
         position_ids,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]
    per_layer_inputs = opts[:per_layer_inputs]

    query_norm = &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, name: &2)
    key_norm = &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, name: &2)

    value_norm = fn value, _name ->
      Axon.nx(value, fn x ->
        variance = Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true)
        Nx.multiply(x, Nx.rsqrt(Nx.add(variance, spec.layer_norm_epsilon)))
      end)
    end

    layer_types = spec.layer_types || generate_layer_types(spec.num_blocks)
    first_kv_shared = spec.num_blocks - spec.num_kv_shared_layers

    # Last occurrence of each layer type before first_kv_shared — these become "store" layers
    store_layer_indices =
      Enum.reduce(0..(first_kv_shared - 1), %{}, fn idx, acc ->
        Map.put(acc, Enum.at(layer_types, idx), idx)
      end)

    attention_scale = 1.0

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    initial_state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cache: cache,
      shared_kv: %{}
    }

    outputs =
      Enum.reduce(0..(spec.num_blocks - 1), initial_state, fn idx, state ->
        layer_type = Enum.at(layer_types, idx)
        is_shared = spec.num_kv_shared_layers > 0 and idx >= first_kv_shared
        is_store = not is_shared and Map.get(store_layer_indices, layer_type) == idx

        block_name = join(join(name, "blocks"), idx)
        block_cache = Layers.Decoder.get_block_cache(state.cache, idx)
        block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])

        head_size =
          case layer_type do
            :full_attention -> spec.global_attention_head_size
            :sliding_attention -> spec.attention_head_size
          end

        num_kv_heads =
          case layer_type do
            :full_attention -> spec.num_global_key_value_heads || spec.num_key_value_heads
            :sliding_attention -> spec.num_key_value_heads
          end

        window_size =
          case layer_type do
            :full_attention -> nil
            :sliding_attention -> {spec.attention_window_size, spec.attention_window_size}
          end

        rotary_opts =
          case layer_type do
            :full_attention ->
              [
                position_ids: position_ids,
                max_positions: spec.max_positions,
                base: spec.rotary_embedding_base,
                rotary_dim: trunc(spec.global_attention_head_size * spec.partial_rotary_factor)
              ]

            :sliding_attention ->
              [
                position_ids: position_ids,
                max_positions: spec.max_positions,
                base: spec.rotary_embedding_base_local
              ]
          end

        precomputed_kv = if is_shared, do: Map.get(state.shared_kv, layer_type), else: nil

        {block_hidden_state, block_cache, pre_rope_kv} =
          gemma4_block(
            state.hidden_state,
            block_cache,
            offset,
            idx,
            spec,
            per_layer_inputs,
            precomputed_kv,
            %{
              attention_mask: attention_mask,
              attention_head_mask: block_attention_head_mask,
              rotary_opts: rotary_opts,
              window_size: window_size,
              head_size: head_size,
              num_kv_heads: num_kv_heads,
              query_norm: query_norm,
              key_norm: key_norm,
              value_norm: value_norm,
              attention_scale: attention_scale,
              first_kv_shared: first_kv_shared,
              name: block_name
            }
          )

        updated_shared_kv =
          if is_store,
            do: Map.put(state.shared_kv, layer_type, pre_rope_kv),
            else: state.shared_kv

        new_cache = Layers.Decoder.put_block_cache(state.cache, idx, block_cache)

        %{
          hidden_state: block_hidden_state,
          hidden_states: Layers.append(state.hidden_states, block_hidden_state),
          attentions: Layers.append(state.attentions, Layers.none()),
          cache: new_cache,
          shared_kv: updated_shared_kv
        }
      end)

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, outputs.hidden_state))
  end

  # Builds one Gemma4 decoder block with:
  # - Pre-attention norm, self-attention, post-attention norm, residual
  # - Pre-FFN norm, FFN, post-FFN norm, residual
  # - Optional PLE block
  # - Layer scalar
  # Returns {hidden_state, block_cache, pre_rope_kv} where pre_rope_kv is nil for shared layers
  defp gemma4_block(hidden_state, block_cache, offset, idx, spec, per_layer_inputs, precomputed_kv, opts) do
    name = opts.name

    # 1. Self-attention
    shortcut = hidden_state

    hidden_state =
      Layers.rms_norm(hidden_state,
        name: join(name, "self_attention_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    {hidden_state, block_cache, pre_rope_kv} =
      gemma4_attention(hidden_state, block_cache, offset, spec, precomputed_kv, opts)

    hidden_state =
      Layers.rms_norm(hidden_state,
        name: join(name, "post_attention_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    hidden_state = Axon.add(shortcut, hidden_state)

    # 2. FFN
    shortcut = hidden_state

    hidden_state =
      Layers.rms_norm(hidden_state,
        name: join(name, "pre_ffn_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    intermediate_size =
      if spec.use_double_wide_mlp and idx >= opts.first_kv_shared do
        spec.intermediate_size * 2
      else
        spec.intermediate_size
      end

    hidden_state =
      gated_ffn(hidden_state, intermediate_size, spec.hidden_size,
        name: join(name, "ffn"),
        activation: spec.activation
      )

    hidden_state =
      Layers.rms_norm(hidden_state,
        name: join(name, "post_ffn_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    hidden_state = Axon.add(shortcut, hidden_state)

    # 3. PLE
    hidden_state =
      if per_layer_inputs do
        ple_slice = Axon.nx(per_layer_inputs, fn x -> x[[.., .., idx, ..]] end)
        shortcut_ple = hidden_state

        gated =
          Axon.dense(hidden_state, spec.hidden_size_per_layer_input,
            name: join(name, "per_layer_input_gate"),
            use_bias: false
          )

        gated = Layers.activation(gated, spec.activation)
        gated = Axon.multiply(gated, ple_slice)

        gated =
          Axon.dense(gated, spec.hidden_size,
            name: join(name, "per_layer_projection"),
            use_bias: false
          )

        gated =
          Layers.rms_norm(gated,
            name: join(name, "post_per_layer_input_norm"),
            epsilon: spec.layer_norm_epsilon
          )

        Axon.add(shortcut_ple, gated)
      else
        hidden_state
      end

    # 4. Layer scalar
    hidden_state =
      Axon.layer(
        fn hidden_state, scalar, _opts ->
          Nx.multiply(hidden_state, Nx.reshape(scalar, {}))
        end,
        [
          hidden_state,
          Axon.param("layer_scalar", fn _ -> {1} end, initializer: Axon.Initializers.ones())
        ],
        name: join(name, "layer_scalar_op")
      )

    {hidden_state, block_cache, pre_rope_kv}
  end

  # Builds self-attention for one Gemma4 block.
  # Non-shared layers: compute Q/K/V, apply RoPE to Q+K, GQA-expand, return post-RoPE expanded K/V.
  # Shared layers: compute Q only, apply RoPE to Q only, reuse stored post-RoPE K/V from store layer.
  # Returns {attention_output, block_cache, storable_kv}.
  defp gemma4_attention(hidden_state, block_cache, offset, spec, precomputed_kv, opts) do
    name = join(opts.name, "self_attention")

    head_size = opts.head_size
    num_kv_heads = opts.num_kv_heads
    num_q_heads = spec.num_attention_heads
    inner_size = num_q_heads * head_size
    inner_kv_size = num_kv_heads * head_size

    rotary_opts = opts.rotary_opts
    position_ids = rotary_opts[:position_ids]

    rotary_call_opts =
      rotary_opts
      |> Keyword.delete(:position_ids)
      |> Keyword.put(:name, join(name, "rotary_embedding"))

    # Q projection + split heads + Q-norm (always computed)
    query =
      hidden_state
      |> Axon.dense(inner_size, name: join(name, "query"), use_bias: spec.use_attention_bias)
      |> Layers.split_heads(num_q_heads)
      |> opts.query_norm.(join(name, "query_norm"))

    {query, key, value, storable_kv} =
      if precomputed_kv do
        # Shared layer: K/V are already post-RoPE, post-GQA from store layer.
        # Apply RoPE to Q only by passing stored key through and discarding the re-rotated key.
        {stored_key, stored_value} = precomputed_kv

        {rotated_query, _discarded} =
          Layers.rotary_embedding(
            query,
            stored_key,
            position_ids,
            opts.attention_mask,
            head_size,
            rotary_call_opts
          )

        {rotated_query, stored_key, stored_value, nil}
      else
        # Non-shared layer: compute K/V projections + norms
        key =
          hidden_state
          |> Axon.dense(inner_kv_size,
            name: join(name, "key"),
            use_bias: spec.use_attention_bias
          )
          |> Layers.split_heads(num_kv_heads)
          |> opts.key_norm.(join(name, "key_norm"))

        value =
          hidden_state
          |> Axon.dense(inner_kv_size,
            name: join(name, "value"),
            use_bias: spec.use_attention_bias
          )
          |> Layers.split_heads(num_kv_heads)
          |> opts.value_norm.(join(name, "value_norm"))

        # Apply RoPE to both Q and K
        {rotated_query, rotated_key} =
          Layers.rotary_embedding(
            query,
            key,
            position_ids,
            opts.attention_mask,
            head_size,
            rotary_call_opts
          )

        # GQA: expand K/V heads to match Q heads
        num_kv_groups = div(num_q_heads, num_kv_heads)

        expanded_key =
          if num_kv_groups > 1,
            do: Layers.repeat_interleave(rotated_key, num_kv_groups, axis: 2),
            else: rotated_key

        expanded_value =
          if num_kv_groups > 1,
            do: Layers.repeat_interleave(value, num_kv_groups, axis: 2),
            else: value

        # Storable: post-RoPE, post-GQA (matches Python's shared_kv_states)
        {rotated_query, expanded_key, expanded_value, {expanded_key, expanded_value}}
      end

    # KV cache update
    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    {key, value, self_attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, self_attention_cache, offset)

    # Scaled dot-product attention
    {attention_output, _weights} =
      Layers.attention(
        query,
        key,
        value,
        opts.attention_mask,
        opts.attention_head_mask,
        Layers.none(),
        offset,
        scale: opts.attention_scale,
        causal: true,
        window_size: opts.window_size
      )

    # Output projection
    attention_output =
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size,
        name: join(name, "output"),
        use_bias: spec.use_attention_bias
      )

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {attention_output, block_cache, storable_kv}
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    gate = Axon.dense(hidden_state, intermediate_size, name: join(name, "gate"), use_bias: false)

    hidden_state = Axon.multiply(intermediate, Layers.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    logits =
      Layers.dense_transposed(hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output")
      )

    cap = spec.final_logit_softcapping

    if cap do
      Axon.nx(logits, fn x ->
        x
        |> Nx.divide(cap)
        |> Nx.tanh()
        |> Nx.multiply(cap)
      end)
    else
      logits
    end
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  # Generate layer_types fallback: every 5th layer uses full attention
  defp generate_layer_types(num_blocks) do
    Enum.map(0..(num_blocks - 1), fn i ->
      if rem(i + 1, 5) == 0 do
        :full_attention
      else
        :sliding_attention
      end
    end)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      data = data["text_config"] || data
      rope_params = data["rope_parameters"] || %{}
      full_attention_rope = rope_params["full_attention"] || %{}
      sliding_attention_rope = rope_params["sliding_attention"] || %{}

      data =
        data
        |> Map.put_new("rope_theta", full_attention_rope["rope_theta"] || 1_000_000)
        |> Map.put_new("rope_local_base_freq", sliding_attention_rope["rope_theta"] || 10_000)
        |> Map.put_new(
          "partial_rotary_factor",
          full_attention_rope["partial_rotary_factor"] || 1.0
        )

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          num_global_key_value_heads: {"num_global_key_value_heads", optional(number())},
          attention_head_size: {"head_dim", number()},
          global_attention_head_size: {"global_head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_activation", activation()},
          use_attention_bias: {"attention_bias", boolean()},
          rotary_embedding_base: {"rope_theta", number()},
          rotary_embedding_base_local: {"rope_local_base_freq", number()},
          partial_rotary_factor: {"partial_rotary_factor", number()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          attention_window_size: {"sliding_window", optional(number())},
          layer_types:
            {"layer_types",
             list(
               mapping(%{
                 "sliding_attention" => :sliding_attention,
                 "full_attention" => :full_attention
               })
             )},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          final_logit_softcapping: {"final_logit_softcapping", optional(number())},
          hidden_size_per_layer_input: {"hidden_size_per_layer_input", number()},
          vocab_size_per_layer_input: {"vocab_size_per_layer_input", number()},
          num_kv_shared_layers: {"num_kv_shared_layers", number()},
          use_double_wide_mlp: {"use_double_wide_mlp", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
      def params_mapping(spec) do
        %{
          "embedder.token_embedding" => "model.language_model.embed_tokens",
          # PLE global weights
          "embedder.token_embedding_per_layer" => "model.language_model.embed_tokens_per_layer",
          "per_layer_model_projection" => "model.language_model.per_layer_model_projection",
          "per_layer_projection_norm" => "model.language_model.per_layer_projection_norm",
          # PLE per-layer weights
          "decoder.blocks.{n}.per_layer_input_gate" =>
            "model.language_model.layers.{n}.per_layer_input_gate",
          "decoder.blocks.{n}.per_layer_projection" =>
            "model.language_model.layers.{n}.per_layer_projection",
          "decoder.blocks.{n}.post_per_layer_input_norm" =>
            "model.language_model.layers.{n}.post_per_layer_input_norm",
          # Layer scalar
          "decoder.blocks.{n}.layer_scalar_op" => "model.language_model.layers.{n}",
          "decoder.blocks.{n}.layer_scalar_op.layer_scalar" =>
            "model.language_model.layers.{n}.layer_scalar",
          # Attention projections
          "decoder.blocks.{n}.self_attention.query" =>
            "model.language_model.layers.{n}.self_attn.q_proj",
          "decoder.blocks.{n}.self_attention.key" =>
            "model.language_model.layers.{n}.self_attn.k_proj",
          "decoder.blocks.{n}.self_attention.value" =>
            "model.language_model.layers.{n}.self_attn.v_proj",
          "decoder.blocks.{n}.self_attention.output" =>
            "model.language_model.layers.{n}.self_attn.o_proj",
          # QK-norm
          "decoder.blocks.{n}.self_attention.query_norm" =>
            "model.language_model.layers.{n}.self_attn.q_norm",
          "decoder.blocks.{n}.self_attention.key_norm" =>
            "model.language_model.layers.{n}.self_attn.k_norm",
          # Layer norms
          "decoder.blocks.{n}.self_attention_norm" =>
            "model.language_model.layers.{n}.input_layernorm",
          "decoder.blocks.{n}.post_attention_norm" =>
            "model.language_model.layers.{n}.post_attention_layernorm",
          # FFN layer norms
          "decoder.blocks.{n}.pre_ffn_norm" =>
            "model.language_model.layers.{n}.pre_feedforward_layernorm",
          "decoder.blocks.{n}.post_ffn_norm" =>
            "model.language_model.layers.{n}.post_feedforward_layernorm",
          # FFN projections
          "decoder.blocks.{n}.ffn.gate" => "model.language_model.layers.{n}.mlp.gate_proj",
          "decoder.blocks.{n}.ffn.intermediate" => "model.language_model.layers.{n}.mlp.up_proj",
          "decoder.blocks.{n}.ffn.output" => "model.language_model.layers.{n}.mlp.down_proj",
          # Output
          "output_norm" => "model.language_model.norm",
          "language_modeling_head.output" =>
            if(spec.tie_word_embeddings,
              do: "model.language_model.embed_tokens",
              else: "lm_head"
            ),
          "sequence_classification_head.output" => "score"
        }
      end
  end
end
