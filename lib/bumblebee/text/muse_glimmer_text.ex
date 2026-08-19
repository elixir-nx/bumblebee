defmodule Bumblebee.Text.MuseGlimmerText do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 202_048,
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
        default: 6656,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 19_968,
        doc: "the dimensionality of intermediate layers"
      ],
      attention_head_size: [
        default: 128,
        doc: "the size of the key, value, and query projection per attention head"
      ],
      num_blocks: [
        default: 52,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 2,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      block_types: [
        default: nil,
        doc: """
        a list with the attention type of each block, either `:full_attention` or
        `:sliding_attention`. When `nil`, every fourth block counted backwards from the last
        one uses full attention and the remaining blocks use sliding window attention
        """
      ],
      attention_window_size: [
        default: 2048,
        doc: "the size of the attention window for blocks using sliding window attention"
      ],
      rotary_embedding_base: [
        default: 500_000,
        doc: "base for computing rotary embedding frequency"
      ],
      block_rotary_embedding_bases: [
        default: nil,
        doc: """
        a list with the rotary embedding base for each block, where `0` means that the block
        uses no positional embedding (NoPE). When `nil`, the blocks using full attention use
        no positional embedding and the remaining blocks use `:rotary_embedding_base`
        """
      ],
      attention_scale_factor: [
        default: 3.87,
        doc: """
        an additional multiplier applied to the query, on top of the usual attention scaling.
        Together with the query and key normalization it sets the target scale of the attention
        logits
        """
      ],
      output_multiplier: [
        default: 0.19611613513818404,
        doc: "the constant that the language modeling logits are multiplied by"
      ],
      logits_softcapping: [
        default: 20.0,
        doc: "the value used for soft capping the language modeling logits"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by RMS normalization layers"
      ],
      post_norm_epsilon: [
        default: 1.0e-8,
        doc: """
        the epsilon used by the RMS normalization layers applied to the output of attention
        and feed-forward blocks, before the shortcut connection
        """
      ],
      use_attention_bias: [
        default: false,
        doc: "whether to use bias in the query, key, value and output projections"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      tie_word_embeddings: [
        default: false,
        doc: "whether to tie input and output embedding weights"
      ],
      params_prefix: [
        default: nil,
        doc: nil
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: nil)

  @moduledoc """
  The text model of the Muse Glimmer model family.

  Muse Glimmer is a multimodal model and this module implements its text
  decoder. The decoder alternates between three blocks with sliding window
  attention using rotary embedding and one block with full attention and
  no positional embedding. Attention uses gated grouped-query attention,
  with normalized query and key.

  ## Architectures

    * `:base` - plain Muse Glimmer text model without any head on top

    * `:for_causal_language_modeling` - Muse Glimmer text model with a
      language modeling head. The head returns logits for each token in
      the original sequence

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

  import Nx.Defn
  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

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
    embeddings = embedder(inputs["input_ids"], inputs["input_embeddings"], spec, name: "embedder")

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

    embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "token_embedding")
        )
      end

    # The embeddings are normalized with a weight-less RMS normalization
    scaleless_rms_norm(embeddings, spec.layer_norm_epsilon)
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

    block_types = block_types(spec)
    rotary_embedding_bases = block_rotary_embedding_bases(spec, block_types)

    attention_window_size = fn idx ->
      case Enum.at(block_types, idx) do
        :full_attention -> nil
        # The window includes the current position, so the maximum
        # distance to an attended position is one less
        :sliding_attention -> {spec.attention_window_size - 1, 0}
      end
    end

    rotary_embedding = fn idx ->
      case Enum.at(rotary_embedding_bases, idx) do
        base when base in [0, 0.0, nil] ->
          nil

        base ->
          [
            position_ids: position_ids,
            max_positions: spec.max_positions,
            base: base
          ]
      end
    end

    query_norm = fn hidden_state, _name ->
      scaleless_rms_norm(hidden_state, spec.layer_norm_epsilon)
    end

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      kernel_initializer: kernel_initializer(spec),
      query_use_bias: spec.use_attention_bias,
      key_use_bias: spec.use_attention_bias,
      value_use_bias: spec.use_attention_bias,
      output_use_bias: spec.use_attention_bias,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      causal: true,
      block_type: &block_impl(&1, &2, &3, spec),
      layer_norm:
        &Layers.rms_norm(&1,
          shift: 1.0,
          epsilon: spec.layer_norm_epsilon,
          upcast: :all,
          name: &2
        ),
      ffn:
        &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
          name: &2,
          activation: spec.activation
        ),
      rotary_embedding: rotary_embedding,
      attention_window_size: attention_window_size,
      attention_scale: spec.attention_scale_factor / :math.sqrt(spec.attention_head_size),
      query_norm: query_norm,
      key_norm: query_norm,
      output_gate_activation: :sigmoid,
      name: join(name, "blocks")
    )
  end

  # Muse Glimmer normalizes the output of both the attention and the
  # feed-forward network before adding the shortcut connection
  defp block_impl(hidden_state, steps, name, spec) do
    shortcut = hidden_state

    {hidden_state, attention_info} =
      hidden_state
      |> steps.self_attention_norm.()
      |> steps.self_attention.()

    hidden_state =
      hidden_state
      |> Layers.rms_norm(
        shift: 1.0,
        name: join(name, "post_attention_norm"),
        epsilon: spec.post_norm_epsilon,
        upcast: :all
      )
      |> Axon.add(shortcut)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> Layers.rms_norm(
        shift: 1.0,
        name: join(name, "pre_ffn_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )
      |> steps.ffn.()
      |> Layers.rms_norm(
        shift: 1.0,
        name: join(name, "post_ffn_norm"),
        epsilon: spec.post_norm_epsilon,
        upcast: :all
      )
      |> Axon.add(shortcut)

    {_hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn _ ->
        raise "cross attention not supported"
      end)

    {hidden_state, attention_info, cross_attention_info}
  end

  defp block_types(spec) do
    spec.block_types ||
      for idx <- 0..(spec.num_blocks - 1) do
        if rem(spec.num_blocks - 1 - idx, 4) == 0, do: :full_attention, else: :sliding_attention
      end
  end

  defp block_rotary_embedding_bases(spec, block_types) do
    spec.block_rotary_embedding_bases ||
      for type <- block_types do
        if type == :full_attention, do: 0, else: spec.rotary_embedding_base
      end
  end

  # RMS normalization without a learnable scale
  defp scaleless_rms_norm(hidden_state, epsilon) do
    Axon.layer(&scaleless_rms_norm_impl/2, [hidden_state], epsilon: epsilon)
  end

  defnp scaleless_rms_norm_impl(hidden_state, opts \\ []) do
    opts = keyword!(opts, [:epsilon, mode: :inference])

    type = Nx.type(hidden_state)
    hidden_state = Nx.as_type(hidden_state, :f32)

    variance = Nx.mean(Nx.pow(hidden_state, 2), axes: [-1], keep_axes: true)

    hidden_state
    |> Nx.multiply(Nx.rsqrt(variance + opts[:epsilon]))
    |> Nx.as_type(type)
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

    hidden_state
    |> Layers.dense_transposed(spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.nx(fn logits ->
      logits = Nx.multiply(logits, spec.output_multiplier)

      case spec.logits_softcapping do
        nil ->
          logits

        cap ->
          logits |> Nx.divide(cap) |> Nx.tanh() |> Nx.multiply(cap)
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

      data = data |> Shared.text_config() |> Shared.normalize_rope_options()

      block_type_converter =
        list(
          mapping(%{
            "full_attention" => :full_attention,
            "sliding_attention" => :sliding_attention
          })
        )

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          activation: {"hidden_activation", activation()},
          block_types: {"layer_types", optional(block_type_converter)},
          attention_window_size: {"sliding_window", number()},
          rotary_embedding_base: {"rope_theta", number()},
          block_rotary_embedding_bases: {"layer_rope_theta", optional(list(number()))},
          attention_scale_factor: {"qk_scale_factor", number()},
          output_multiplier: {"output_multiplier", number()},
          logits_softcapping: {"final_logit_softcapping", optional(number())},
          use_attention_bias: {"attention_bias", boolean()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          post_norm_epsilon: {"post_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec) ++ params_prefix

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention.output_gate" => "model.layers.{n}.self_attn.gate_proj",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.post_attention_norm" => "model.layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.pre_ffn_norm" => "model.layers.{n}.pre_feedforward_layernorm",
        "decoder.blocks.{n}.post_ffn_norm" => "model.layers.{n}.post_feedforward_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head")
      }
      |> Bumblebee.Shared.replace_params_mapping_source_prefix(spec.params_prefix)
    end
  end
end
