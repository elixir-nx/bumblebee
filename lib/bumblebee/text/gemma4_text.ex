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
        default: 262_144,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 5376,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 21_504,
        doc: "the dimensionality of intermediate layers"
      ],
      num_blocks: [
        default: 60,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 16,
        doc: "the number of key value heads in the blocks using sliding window attention"
      ],
      num_global_key_value_heads: [
        default: 4,
        doc: "the number of key value heads in the blocks using full attention"
      ],
      attention_head_size: [
        default: 256,
        doc:
          "the size of the projection per attention head in the blocks using sliding window attention"
      ],
      global_attention_head_size: [
        default: 512,
        doc: "the size of the projection per attention head in the blocks using full attention"
      ],
      block_types: [
        default: nil,
        doc: """
        a list with the attention type of each block, either `:full_attention` or
        `:sliding_attention`. When `nil`, every sixth block uses full attention
        """
      ],
      attention_window_size: [
        default: 1024,
        doc: "the size of the attention window for blocks using sliding window attention"
      ],
      share_key_value: [
        default: true,
        doc: """
        whether the blocks using full attention use the key projection as the value projection,
        rather than having a separate one
        """
      ],
      num_experts: [
        default: nil,
        doc: """
        the number of experts in each mixture-of-experts block. When `nil`, the blocks use a
        regular feed-forward network only
        """
      ],
      num_experts_per_token: [
        default: 8,
        doc: "the number of experts that each token is routed to"
      ],
      moe_intermediate_size: [
        default: 704,
        doc: "the dimensionality of intermediate layers in each mixture-of-experts expert"
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency in the sliding window blocks"
      ],
      global_rotary_embedding_base: [
        default: 1_000_000,
        doc: "base for computing rotary embedding frequency in the full attention blocks"
      ],
      global_rotary_embedding_partial_factor: [
        default: 0.25,
        doc: """
        the fraction of the head that rotary embedding is applied to in the full attention
        blocks
        """
      ],
      logits_softcapping: [
        default: 30.0,
        doc: "the value used for soft capping the language modeling logits"
      ],
      use_attention_bias: [
        default: false,
        doc: "whether to use bias in the query, key, value and output projections"
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
        default: true,
        doc: "whether to tie input and output embedding weights"
      ],
      params_prefix: [
        default: nil,
        doc: nil
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 0)

  @moduledoc """
  The text model of the Gemma 4 model family.

  Gemma 4 is a multimodal model and this module implements its text
  decoder. The blocks alternate between sliding window and full attention,
  and the two kinds differ in the number of key-value heads, the head
  size and the rotary embedding. Query, key and value are normalized and
  the attention logits are not scaled. Depending on the configuration,
  the blocks may additionally include a mixture-of-experts network,
  computed in parallel with the regular feed-forward network.

  ## Architectures

    * `:base` - plain Gemma 4 text model without any head on top

    * `:for_causal_language_modeling` - Gemma 4 text model with a language
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
    block_types = block_types(spec)

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      attention_head_size: fn idx -> head_size(spec, Enum.at(block_types, idx)) end,
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
        inputs["input_ids"]
        |> Axon.embedding(spec.vocab_size, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: "embedder.token_embedding"
        )
        |> Axon.nx(fn embeddings ->
          Nx.multiply(embeddings, Nx.sqrt(Nx.tensor(spec.hidden_size, type: Nx.type(embeddings))))
        end)
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
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

  defp decoder(hidden_state, position_ids, attention_mask, cache, spec, opts) do
    name = opts[:name]

    block_types = block_types(spec)

    attention_window_size = fn idx ->
      case Enum.at(block_types, idx) do
        :full_attention ->
          nil

        :sliding_attention ->
          # The window includes the current position, so the maximum
          # distance to an attended position is one less
          {spec.attention_window_size - 1, 0}
      end
    end

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      attention_mask: attention_mask,
      cache: cache,
      causal: true,
      attention_window_size: attention_window_size,
      layer_norm: &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, upcast: :all, name: &2),
      block_type: &block_impl(&1, &2, &3, spec),
      attention: fn idx ->
        &attention(&1, &2, spec, position_ids, Enum.at(block_types, idx))
      end,
      ffn: fn _idx ->
        &ffn(&1, spec, name: &2)
      end,
      name: join(name, "blocks")
    )
  end

  # Gemma 4 normalizes the output of both the attention and the
  # feed-forward network before adding the shortcut connection, and
  # scales the block output
  defp block_impl(hidden_state, steps, name, spec) do
    shortcut = hidden_state

    {hidden_state, attention_info} =
      hidden_state
      |> steps.self_attention_norm.()
      |> steps.self_attention.()

    hidden_state =
      hidden_state
      |> Layers.rms_norm(
        name: join(name, "post_attention_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )
      |> Axon.add(shortcut)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> Layers.rms_norm(
        name: join(name, "pre_ffn_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )
      |> steps.ffn.()

    hidden_state =
      if spec.num_experts do
        # The mixture-of-experts network runs in parallel with the
        # feed-forward network, taking the block input as its input
        dense =
          Layers.rms_norm(hidden_state,
            name: join(name, "ffn.post_norm"),
            epsilon: spec.layer_norm_epsilon,
            upcast: :all
          )

        sparse =
          shortcut
          |> Layers.rms_norm(
            name: join(name, "moe.pre_norm"),
            epsilon: spec.layer_norm_epsilon,
            upcast: :all
          )
          |> moe(shortcut, spec, name: join(name, "moe"))
          |> Layers.rms_norm(
            name: join(name, "moe.post_norm"),
            epsilon: spec.layer_norm_epsilon,
            upcast: :all
          )

        Axon.add(dense, sparse)
      else
        hidden_state
      end

    hidden_state =
      hidden_state
      |> Layers.rms_norm(
        name: join(name, "post_ffn_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )
      |> Axon.add(shortcut)
      |> block_scale(join(name, "scale"))

    {_hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn _ ->
        raise "cross attention not supported"
      end)

    {hidden_state, attention_info, cross_attention_info}
  end

  defp attention(hidden_state, opts, spec, position_ids, block_type) do
    name = opts[:name]

    num_heads = spec.num_attention_heads
    num_key_value_heads = num_key_value_heads(spec, block_type)
    head_size = head_size(spec, block_type)
    share_key_value? = spec.share_key_value and block_type == :full_attention

    query =
      hidden_state
      |> Axon.dense(num_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query"),
        use_bias: spec.use_attention_bias
      )
      |> Layers.split_heads(num_heads)
      |> Layers.rms_norm(
        name: join(name, "query_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    key_projection =
      hidden_state
      |> Axon.dense(num_key_value_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key"),
        use_bias: spec.use_attention_bias
      )
      |> Layers.split_heads(num_key_value_heads)

    key =
      Layers.rms_norm(key_projection,
        name: join(name, "key_norm"),
        epsilon: spec.layer_norm_epsilon,
        upcast: :all
      )

    # In the blocks using full attention the key projection doubles as
    # the value projection
    value =
      if share_key_value? do
        key_projection
      else
        hidden_state
        |> Axon.dense(num_key_value_heads * head_size,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "value"),
          use_bias: spec.use_attention_bias
        )
        |> Layers.split_heads(num_key_value_heads)
      end

    value = scaleless_rms_norm(value, spec.layer_norm_epsilon)

    {rotary_base, scaling_strategy} =
      case block_type do
        :full_attention ->
          {spec.global_rotary_embedding_base,
           %{
             type: :proportional,
             partial_rotary_factor: spec.global_rotary_embedding_partial_factor,
             factor: 1.0
           }}

        :sliding_attention ->
          {spec.rotary_embedding_base, nil}
      end

    {query, key} =
      Layers.rotary_embedding(query, key, position_ids, opts[:attention_mask], head_size,
        name: join(name, "rotary_embedding"),
        max_positions: spec.max_positions,
        base: rotary_base,
        scaling_strategy: scaling_strategy
      )

    num_key_value_groups = div(num_heads, num_key_value_heads)
    key = repeat_states(key, num_key_value_groups)
    value = repeat_states(value, num_key_value_groups)

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(
        key,
        value,
        opts[:attention_cache],
        opts[:offset]
      )

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        opts[:attention_mask],
        opts[:attention_head_mask],
        Layers.none(),
        opts[:offset],
        causal: opts[:causal],
        window_size: opts[:attention_window_size],
        # Query and key are normalized, so the logits are not scaled
        scale: 1.0
      )

    attention_output =
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output"),
        use_bias: spec.use_attention_bias
      )

    {attention_output, attention_weights, attention_cache, Layers.none()}
  end

  defp repeat_states(state, 1), do: state
  defp repeat_states(state, times), do: Layers.repeat_interleave(state, times, axis: 2)

  defp ffn(hidden_state, spec, opts) do
    name = opts[:name]

    intermediate =
      Axon.dense(hidden_state, spec.intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    gate =
      Axon.dense(hidden_state, spec.intermediate_size, name: join(name, "gate"), use_bias: false)

    hidden_state = Axon.multiply(intermediate, Layers.activation(gate, spec.activation))

    Axon.dense(hidden_state, spec.hidden_size, name: join(name, "output"), use_bias: false)
  end

  defp moe(hidden_state, router_input, spec, opts) do
    name = opts[:name]

    weights = router(router_input, spec, name: join(name, "router"))

    Moe.experts(hidden_state, weights,
      num_experts: spec.num_experts,
      hidden_size: spec.hidden_size,
      intermediate_size: spec.moe_intermediate_size,
      activation: spec.activation,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "experts")
    )
  end

  # The router normalizes and rescales its input, and applies a learnt
  # per-expert scale to the resulting weights
  defp router(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state = scaleless_rms_norm(hidden_state, spec.layer_norm_epsilon)

    scale =
      Axon.param("scale", fn shape -> {elem(shape, tuple_size(shape) - 1)} end,
        initializer: :ones
      )

    kernel =
      Axon.param("kernel", fn shape -> {elem(shape, tuple_size(shape) - 1), spec.num_experts} end,
        initializer: kernel_initializer(spec)
      )

    expert_scale =
      Axon.param("expert_scale", fn _ -> {spec.num_experts} end, initializer: :ones)

    Axon.layer(&router_impl/5, [hidden_state, scale, kernel, expert_scale],
      name: name,
      op_name: :moe_router,
      num_experts_per_token: spec.num_experts_per_token,
      input_scale: 1.0 / :math.sqrt(spec.hidden_size)
    )
  end

  defnp router_impl(hidden_state, scale, kernel, expert_scale, opts \\ []) do
    opts = keyword!(opts, [:num_experts_per_token, :input_scale, mode: :inference])

    hidden_state = Nx.as_type(hidden_state, :f32)

    hidden_state = hidden_state * Nx.as_type(scale, :f32) * opts[:input_scale]

    logits = Nx.dot(hidden_state, [-1], Nx.as_type(kernel, :f32), [0])

    probabilities = Axon.Activations.softmax(logits, axis: -1)

    order = Nx.argsort(probabilities, axis: -1, direction: :desc, stable: true)
    rank = Nx.argsort(order, axis: -1, stable: true)
    mask = Nx.as_type(Nx.less(rank, opts[:num_experts_per_token]), :f32)

    weights = probabilities * mask
    weights = weights / Nx.sum(weights, axes: [-1], keep_axes: true)

    weights * Nx.as_type(expert_scale, :f32)
  end

  defp block_scale(hidden_state, name) do
    scale = Axon.param("scale", fn _ -> {1} end, initializer: :ones)

    Axon.layer(
      fn hidden_state, scale, _opts -> Nx.multiply(hidden_state, scale) end,
      [hidden_state, scale],
      name: name,
      op_name: :block_scale
    )
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

  defp block_types(spec) do
    spec.block_types ||
      for idx <- 0..(spec.num_blocks - 1) do
        if rem(idx + 1, 6) == 0, do: :full_attention, else: :sliding_attention
      end
  end

  defp head_size(spec, :full_attention), do: spec.global_attention_head_size
  defp head_size(spec, :sliding_attention), do: spec.attention_head_size

  defp num_key_value_heads(spec, :full_attention), do: spec.num_global_key_value_heads
  defp num_key_value_heads(spec, :sliding_attention), do: spec.num_key_value_heads

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.dense_transposed(spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.nx(fn logits ->
      case spec.logits_softcapping do
        nil -> logits
        cap -> logits |> Nx.divide(cap) |> Nx.tanh() |> Nx.multiply(cap)
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

      if data["hidden_size_per_layer_input"] not in [nil, 0] do
        raise ArgumentError,
              "Gemma 4 checkpoints with per-layer embeddings are not supported yet" <>
                " (hidden_size_per_layer_input: #{inspect(data["hidden_size_per_layer_input"])})"
      end

      if data["num_kv_shared_layers"] not in [nil, 0] do
        raise ArgumentError,
              "Gemma 4 checkpoints with key-value sharing across blocks are not supported yet" <>
                " (num_kv_shared_layers: #{inspect(data["num_kv_shared_layers"])})"
      end

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
          block_types: {"layer_types", optional(block_type_converter)},
          attention_window_size: {"sliding_window", number()},
          share_key_value: {"attention_k_eq_v", boolean()},
          num_experts: {"num_experts", optional(number())},
          num_experts_per_token: {"top_k_experts", optional(number())},
          moe_intermediate_size: {"moe_intermediate_size", optional(number())},
          activation: {"hidden_activation", activation()},
          logits_softcapping: {"final_logit_softcapping", optional(number())},
          use_attention_bias: {"attention_bias", boolean()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec) ++ params_prefix

      opts =
        case data["rope_parameters"] do
          %{} = parameters -> Keyword.merge(opts, rotary_options(parameters))
          _other -> opts
        end

      opts = Keyword.merge(opts, full_attention_options(data))

      # The mixture-of-experts blocks are only enabled explicitly
      opts =
        if data["enable_moe_block"] == false do
          Keyword.put(opts, :num_experts, nil)
        else
          opts
        end

      @for.config(spec, opts)
    end

    # The blocks using full attention have a different head size and
    # number of key-value heads, which the configuration expresses either
    # as per-block overrides, or as separate top-level keys
    defp full_attention_options(data) do
      override =
        case data["per_layer_config"] do
          %{} = per_layer_config -> per_layer_config |> Map.values() |> List.first() || %{}
          _other -> %{}
        end

      head_size = override["head_dim"] || data["global_head_dim"] || data["head_dim"]

      num_key_value_heads =
        override["num_key_value_heads"] || data["num_global_key_value_heads"] ||
          data["num_key_value_heads"]

      []
      |> put_option(:global_attention_head_size, head_size)
      |> put_option(:num_global_key_value_heads, num_key_value_heads)
    end

    # Gemma 4 configures rotary embedding separately for each block type
    defp rotary_options(parameters) do
      sliding = parameters["sliding_attention"] || %{}
      full = parameters["full_attention"] || %{}

      []
      |> put_option(:rotary_embedding_base, sliding["rope_theta"])
      |> put_option(:global_rotary_embedding_base, full["rope_theta"])
      |> put_option(
        :global_rotary_embedding_partial_factor,
        full["partial_rotary_factor"]
      )
    end

    defp put_option(opts, _key, nil), do: opts
    defp put_option(opts, key, value), do: Keyword.put(opts, key, value)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.Layers.Moe

    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention.query_norm" => "model.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" => "model.layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.post_attention_norm" => "model.layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.pre_ffn_norm" => "model.layers.{n}.pre_feedforward_layernorm",
        "decoder.blocks.{n}.post_ffn_norm" => "model.layers.{n}.post_feedforward_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.ffn.post_norm" => "model.layers.{n}.post_feedforward_layernorm_1",
        "decoder.blocks.{n}.moe.pre_norm" => "model.layers.{n}.pre_feedforward_layernorm_2",
        "decoder.blocks.{n}.moe.post_norm" => "model.layers.{n}.post_feedforward_layernorm_2",
        "decoder.blocks.{n}.scale" => %{
          "scale" => {
            [{"model.layers.{n}", "layer_scalar"}],
            fn [scale] -> Nx.reshape(scale, {1}) end
          }
        },
        "decoder.blocks.{n}.moe.router" => %{
          "kernel" => {
            [{"model.layers.{n}.router.proj", "weight"}],
            fn [kernel] -> Nx.transpose(kernel) end
          },
          "scale" => {
            [{"model.layers.{n}.router", "scale"}],
            fn [scale] -> scale end
          },
          "expert_scale" => {
            [{"model.layers.{n}.router", "per_expert_scale"}],
            fn [scale] -> scale end
          }
        },
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head")
      }
      |> Map.merge(expert_params(spec))
      |> Bumblebee.Shared.replace_params_mapping_source_prefix(spec.params_prefix)
    end

    defp expert_params(%{num_experts: nil}), do: %{}

    defp expert_params(spec) do
      experts = "model.layers.{n}.experts"

      refs = fn projection ->
        for idx <- 0..(spec.num_experts - 1) do
          [
            {experts <> ".#{idx}.#{projection}", "weight"},
            {experts, if(projection == "down_proj", do: "down_proj", else: "gate_up_proj")}
          ]
        end
      end

      %{
        "decoder.blocks.{n}.moe.experts" => %{
          "gate_kernel" =>
            {refs.("gate_proj"), &Moe.stack_expert_kernels(&1, 0, spec.moe_intermediate_size)},
          "up_kernel" =>
            {refs.("up_proj"),
             &Moe.stack_expert_kernels(
               &1,
               spec.moe_intermediate_size,
               spec.moe_intermediate_size
             )},
          "down_kernel" => {refs.("down_proj"), &Moe.stack_expert_kernels(&1)}
        }
      }
    end
  end
end
