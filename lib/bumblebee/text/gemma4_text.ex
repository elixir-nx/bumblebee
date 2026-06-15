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
        shift: 1.0,
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
        shift: 1.0,
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

    query_norm = &Layers.rms_norm(&1, shift: 1.0, epsilon: spec.layer_norm_epsilon, name: &2)
    key_norm = &Layers.rms_norm(&1, shift: 1.0, epsilon: spec.layer_norm_epsilon, name: &2)

    value_norm = fn value, _name ->
      Axon.nx(value, fn x ->
        variance = Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true)
        Nx.multiply(x, Nx.rsqrt(Nx.add(variance, spec.layer_norm_epsilon)))
      end)
    end

    layer_types = spec.layer_types || generate_layer_types(spec.num_blocks)

    attention_window_size = fn idx ->
      case Enum.at(layer_types, idx, :sliding_attention) do
        :full_attention -> nil
        :sliding_attention -> {spec.attention_window_size, spec.attention_window_size}
      end
    end

    rotary_embedding = fn idx ->
      case Enum.at(layer_types, idx, :sliding_attention) do
        :full_attention ->
          [
            position_ids: position_ids,
            max_positions: spec.max_positions,
            base: spec.rotary_embedding_base,
            percentage: 1.0,
            rotary_dim:
              trunc(spec.global_attention_head_size * spec.partial_rotary_factor)
          ]

        :sliding_attention ->
          [
            position_ids: position_ids,
            max_positions: spec.max_positions,
            base: spec.rotary_embedding_base_local,
            percentage: 1.0
          ]
      end
    end

    attention_scale = 1.0
    non_double_wide_count = spec.num_blocks - spec.num_kv_shared_layers

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      attention_head_size: fn idx ->
        case Enum.at(layer_types, idx, :sliding_attention) do
          :full_attention -> spec.global_attention_head_size
          :sliding_attention -> spec.attention_head_size
        end
      end,
      attention_scale: attention_scale,
      kernel_initializer: kernel_initializer(spec),
      layer_norm:
        &Layers.rms_norm(&1,
          shift: 1.0,
          name: &2,
          epsilon: spec.layer_norm_epsilon,
          upcast: :all
        ),
      ffn: fn hidden_state, ffn_name ->
        idx =
          ffn_name
          |> String.split(".")
          |> Enum.at(2)
          |> String.to_integer()

        intermediate_size =
          if spec.use_double_wide_mlp and idx >= non_double_wide_count do
            spec.intermediate_size * 2
          else
            spec.intermediate_size
          end

        gated_ffn(hidden_state, intermediate_size, spec.hidden_size,
          name: ffn_name,
          activation: spec.activation
        )
      end,
      block_type: &gemma4_block_impl(&1, &2, &3, spec, per_layer_inputs),
      causal: true,
      rotary_embedding: rotary_embedding,
      attention_window_size: attention_window_size,
      query_norm: query_norm,
      key_norm: key_norm,
      value_norm: value_norm,
      query_use_bias: spec.use_attention_bias,
      key_use_bias: spec.use_attention_bias,
      value_use_bias: spec.use_attention_bias,
      output_use_bias: spec.use_attention_bias,
      name: join(name, "blocks")
    )
  end

  # Custom block implementation for Gemma 4's normalization structure:
  # - Post-attention norm BEFORE residual add
  # - Pre/post FFN norms
  # - PLE applied AFTER attention+MLP as a third residual block
  defp gemma4_block_impl(hidden_state, steps, name, spec, per_layer_inputs) do
    # 1. Self-attention with pre/post norms
    shortcut = hidden_state

    {hidden_state, attention_info} =
      hidden_state
      |> steps.self_attention_norm.()
      |> steps.self_attention.()

    hidden_state =
      Layers.rms_norm(hidden_state,
        shift: 1.0,
        name: join(name, "post_attention_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    hidden_state = Axon.add(shortcut, hidden_state)

    # 2. FFN with pre/post norms
    shortcut = hidden_state

    hidden_state =
      Layers.rms_norm(hidden_state,
        shift: 1.0,
        name: join(name, "pre_ffn_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    hidden_state = steps.ffn.(hidden_state)

    hidden_state =
      Layers.rms_norm(hidden_state,
        shift: 1.0,
        name: join(name, "post_ffn_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    hidden_state = Axon.add(shortcut, hidden_state)

    # 3. PLE: gate hidden_state down to PLE dim, multiply with PLE signal, project back up
    hidden_state =
      if per_layer_inputs do
        idx =
          name
          |> String.split(".")
          |> Enum.at(2)
          |> String.to_integer()

        ple_slice =
          Axon.nx(per_layer_inputs, fn x ->
            x[[.., .., idx, ..]]
          end)

        shortcut_ple = hidden_state

        # Gate: project hidden_state DOWN to PLE dimension
        gated =
          Axon.dense(hidden_state, spec.hidden_size_per_layer_input,
            name: join(name, "per_layer_input_gate"),
            use_bias: false
          )

        # Activation (gelu_approx_tanh, same as FFN)
        gated = Layers.activation(gated, spec.activation)

        # Element-wise multiply with PLE signal
        gated = Axon.multiply(gated, ple_slice)

        # Project back UP to hidden dimension
        gated =
          Axon.dense(gated, spec.hidden_size,
            name: join(name, "per_layer_projection"),
            use_bias: false
          )

        # Normalize
        gated =
          Layers.rms_norm(gated,
            shift: 1.0,
            name: join(name, "post_per_layer_input_norm"),
            epsilon: spec.layer_norm_epsilon
          )

        Axon.add(shortcut_ple, gated)
      else
        hidden_state
      end

    # 4. Layer scalar: multiply output by per-layer learned scalar
    layer_scalar =
      Axon.param("layer_scalar", fn _ -> {1} end, initializer: :ones)

    hidden_state =
      Axon.layer(
        fn hidden_state, scalar, _opts ->
          Nx.multiply(hidden_state, scalar)
        end,
        [hidden_state, layer_scalar],
        name: join(name, "layer_scalar")
      )

    {_hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn _ ->
        raise "cross attention not supported"
      end)

    {hidden_state, attention_info, cross_attention_info}
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
        # PLE (Per-Layer Embeddings) global weights
        "embedder.token_embedding_per_layer" => "model.language_model.embed_tokens_per_layer",
        "per_layer_model_projection" => "model.language_model.per_layer_model_projection",
        "per_layer_projection_norm" => "model.language_model.per_layer_projection_norm",
        "decoder.blocks.{n}.per_layer_input_gate" =>
          "model.language_model.layers.{n}.per_layer_input_gate",
        "decoder.blocks.{n}.per_layer_projection" =>
          "model.language_model.layers.{n}.per_layer_projection",
        "decoder.blocks.{n}.post_per_layer_input_norm" =>
          "model.language_model.layers.{n}.post_per_layer_input_norm",
        # Per-layer scalar
        "decoder.blocks.{n}.layer_scalar" =>
          "model.language_model.layers.{n}",
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
