defmodule Bumblebee.Text.Gemma3 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 262_208,
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
        default: 2304,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 9216,
        doc: "the dimensionality of intermediate layers"
      ],
      attention_head_size: [
        default: 256,
        doc: "the size of the key, value, and query projection per attention head"
      ],
      num_blocks: [
        default: 26,
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
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency"
      ],
      rotary_embedding_scaling_strategy: [
        default: nil,
        doc: """
        scaling configuration for rotary embedding. Currently the supported values are:

          * `%{type: :linear, factor: number()}`

          * `%{type: :dynamic, factor: number()}`

        For more details see https://www.reddit.com/r/LocalLlama/comments/14mrgpr/dynamically_scaled_rope_further_increases
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
      sliding_window: [
        default: 4096,
        doc: "the sliding window size for local attention layers"
      ],
      global_attention_layer_interval: [
        default: 6,
        doc: """
        the interval for global attention layers. In Gemma 3, every Nth layer uses global
        attention while others use local (sliding window) attention. A value of 6 means
        layers 5, 11, 17, 23... use global attention (5:1 local/global ratio)
        """
      ],
      tie_word_embeddings: [
        default: true,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++ Shared.token_options(pad_token_id: 0)

  @moduledoc """
  Gemma 3 model family.

  Gemma 3 is an updated version of the Gemma architecture with several key improvements:

    * Alternating local/global attention (5:1 ratio by default) for better efficiency
    * Larger vocabulary (262K tokens)
    * Extended context length (up to 128K tokens)

  This module also supports FunctionGemma, which is built on Gemma 3 and optimized
  for function calling tasks.

  ## Architectures

    * `:base` - plain Gemma 3 without any head on top

    * `:for_causal_language_modeling` - Gemma 3 with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - Gemma 3 with a sequence
      classification head. The head returns logits corresponding to
      possible classes

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

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

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
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks
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

    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
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

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    # Note: Gemma 3 still normalizes embeddings by sqrt(hidden_size), same as Gemma v1
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

    # QK-norm functions for Gemma 3 (uses shift: 1.0 for (1+weight) formula)
    query_norm = &Layers.rms_norm(&1, shift: 1.0, epsilon: spec.layer_norm_epsilon, name: &2)
    key_norm = &Layers.rms_norm(&1, shift: 1.0, epsilon: spec.layer_norm_epsilon, name: &2)

    # Per-layer attention window size for alternating local/global attention
    # Every global_attention_layer_interval-th layer uses global attention
    attention_window_size = fn idx ->
      if rem(idx + 1, spec.global_attention_layer_interval) == 0 do
        nil
      else
        {spec.sliding_window, spec.sliding_window}
      end
    end

    # Custom block_type function for Gemma 3's unique block structure
    block_type = fn hidden_state, steps, block_name ->
      gemma3_block_impl(hidden_state, steps, block_name, spec)
    end

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      kernel_initializer: kernel_initializer(spec),
      layer_norm:
        &Layers.rms_norm(&1,
          shift: 1.0,
          name: &2,
          epsilon: spec.layer_norm_epsilon,
          upcast: :all
        ),
      ffn:
        &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
          name: &2,
          activation: spec.activation
        ),
      block_type: block_type,
      causal: true,
      rotary_embedding: [
        position_ids: position_ids,
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base,
        scaling_strategy: spec.rotary_embedding_scaling_strategy
      ],
      attention_window_size: attention_window_size,
      query_norm: query_norm,
      key_norm: key_norm,
      query_use_bias: spec.use_attention_bias,
      key_use_bias: spec.use_attention_bias,
      value_use_bias: spec.use_attention_bias,
      output_use_bias: spec.use_attention_bias,
      name: join(name, "blocks")
    )
  end

  # Custom block implementation for Gemma 3's unique normalization structure:
  # - Post-attention norm BEFORE residual add
  # - Pre/post FFN norms
  defp gemma3_block_impl(hidden_state, steps, name, spec) do
    # Pre-attention norm + attention (using provided steps)
    shortcut = hidden_state

    {hidden_state, attention_info} =
      hidden_state
      |> steps.self_attention_norm.()
      |> steps.self_attention.()

    # Post-attention norm BEFORE residual (Gemma 3 specific)
    hidden_state =
      Layers.rms_norm(hidden_state,
        shift: 1.0,
        name: join(name, "post_attention_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    hidden_state = Axon.add(shortcut, hidden_state)

    # FFN with pre/post norms (Gemma 3 specific)
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

    # Handle cross-attention (required by block interface but not used by Gemma 3)
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

    # TODO: Tie lm-head to word embedding as a spec option
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

      scaling_strategy_converter = fn name, value ->
        case value do
          %{"type" => "linear", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :linear, factor: factor}}

          %{"type" => "dynamic", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :dynamic, factor: factor}}

          _other ->
            {:error, "invalid format for #{inspect(name)}, got: #{inspect(value)}"}
        end
      end

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_activation", activation()},
          use_attention_bias: {"attention_bias", boolean()},
          rotary_embedding_base: {"rope_theta", number()},
          rotary_embedding_scaling_strategy:
            {"rope_scaling", optional(scaling_strategy_converter)},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          sliding_window: {"sliding_window", optional(number())},
          tie_word_embeddings: {"tie_word_embeddings", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      # Gemma 3 specific params mapping with QK-norm and extra FFN layer norms
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        # Attention projections
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        # QK-norm (Gemma 3 specific) - uses query_norm/key_norm from shared infrastructure
        "decoder.blocks.{n}.self_attention.query_norm" => "model.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" => "model.layers.{n}.self_attn.k_norm",
        # Layer norms
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.post_attention_norm" => "model.layers.{n}.post_attention_layernorm",
        # FFN layer norms (Gemma 3 specific)
        "decoder.blocks.{n}.pre_ffn_norm" => "model.layers.{n}.pre_feedforward_layernorm",
        "decoder.blocks.{n}.post_ffn_norm" => "model.layers.{n}.post_feedforward_layernorm",
        # FFN projections
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        # Output
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head"),
        "sequence_classification_head.output" => "score"
      }
    end
  end
end
