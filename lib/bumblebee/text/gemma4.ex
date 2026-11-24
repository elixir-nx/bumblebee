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
        default: 2304,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 9216,
        doc: "the dimensionality of intermediate layers"
      ],
      use_double_wide_ffn: [
        default: false,
        doc:
          "whether blocks that reuse key and value states from an earlier block use an intermediate layer twice as wide"
      ],
      attention_head_size: [
        default: 256,
        doc:
          "the size of the key, value, and query projection per attention head in `:sliding_attention` layers"
      ],
      global_attention_head_size: [
        default: 512,
        doc:
          "the size of the key, value, and query projection per attention head in `:full_attention` layers"
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
      global_num_key_value_heads: [
        default: nil,
        doc: """
        the number of key value heads in `:full_attention` layers, when `:attention_key_equals_value`
        is set. Defaults to `:num_key_value_heads`
        """
      ],
      attention_key_equals_value: [
        default: false,
        doc:
          "whether `:full_attention` layers use the key projection as the value projection, without a separate value projection"
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 1_000_000,
        doc: "base for computing rotary embedding frequency for `:full_attention` layers"
      ],
      rotary_embedding_percentage: [
        default: 1.0,
        doc: "the fraction of the head rotated by standard RoPE in full-attention layers"
      ],
      rotary_embedding_scaling_strategy: [
        default: %{type: :proportional, percentage: 0.25, factor: 1.0},
        doc: "the rotary embedding scaling strategy in full-attention layers"
      ],
      rotary_embedding_base_local: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency for `:sliding_attention` layers"
      ],
      rotary_embedding_percentage_local: [
        default: 1.0,
        doc: "the fraction of the head rotated by standard RoPE in sliding-attention layers"
      ],
      rotary_embedding_scaling_strategy_local: [
        default: nil,
        doc: "the rotary embedding scaling strategy in sliding-attention layers"
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
          "the number of tokens each token can attend to in the sliding attention window, including itself. Only used for `:sliding_attention` layers"
      ],
      layer_types: [
        default: nil,
        doc: """
        a list of layer types for each layer, where each element is either `:sliding_attention`
        (local attention with sliding window) or `:full_attention` (global attention). Defaults
        to five `:sliding_attention` layers followed by one `:full_attention` layer, repeated
        """
      ],
      per_layer_input_size: [
        default: 256,
        doc: """
        the dimensionality of the per-layer input, an embedding of each token for each block
        that is added to the block output. Set to `0` to disable per-layer inputs
        """
      ],
      per_layer_input_vocab_size: [
        default: 262_144,
        doc: "the vocabulary size of the per-layer input embedding"
      ],
      num_shared_key_value_blocks: [
        default: 0,
        doc: """
        the number of final blocks that reuse key and value states from the last earlier block
        of the same layer type, instead of computing their own
        """
      ],
      final_logit_softcapping: [
        default: nil,
        doc: "the value to cap output logits at, with `tanh`. No capping when `nil`"
      ],
      tie_word_embeddings: [
        default: true,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 0)

  @moduledoc """
  Gemma 4 model family, the text part.

  Compared to Gemma 3, Gemma 4 has:

    * Larger attention heads in global layers, with rotary embedding
      on a fraction of the frequencies

    * Per-layer inputs: an embedding of each token for each block, mixed
      into the block output through a gate (E2B and E4B)

    * Key and value states of final blocks reused from earlier blocks
      (E2B and E4B)

    * The key projection used as the value projection in global layers
      (31B)

  Mixture-of-experts models, such as 26B-A4B, are not supported yet.

  ## Architectures

    * `:base` - plain Gemma 4 without any head on top

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

    * `"attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored. Per-layer inputs then only have
      their part projected from the embeddings.

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
  def architectures(), do: [:base, :for_causal_language_modeling]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    layer_types = layer_types(spec)
    first_shared = first_shared_block(spec)

    blocks =
      for index <- 0..(spec.num_blocks - 1) do
        self_attention =
          if index < first_shared do
            {num_key_value_heads, head_size} = key_value_shape(spec, Enum.at(layer_types, index))
            zeros = Nx.broadcast(0.0, {batch_size, max_length, num_key_value_heads, head_size})
            %{key: zeros, value: zeros}
          else
            %Axon.None{}
          end

        %{self_attention: self_attention, cross_attention: %Axon.None{}}
      end

    %Layers.Decoder.Cache{
      blocks: List.to_tuple(blocks),
      offset: Nx.tensor(0),
      attention_mask: Nx.broadcast(0, {batch_size, max_length})
    }
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
    embeddings =
      embedder(inputs["input_ids"], inputs["input_embeddings"], spec, name: "embedder")

    per_layer_inputs =
      if spec.per_layer_input_size > 0 do
        per_layer_inputs(inputs["input_ids"], inputs["input_embeddings"], embeddings, spec,
          name: "per_layer_inputs"
        )
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      decoder(
        embeddings,
        per_layer_inputs,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state = norm(decoder_outputs.hidden_state, spec, "output_norm")

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
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
      |> scale(:math.sqrt(spec.hidden_size))
    end
  end

  # Per-layer inputs combine an embedding of each token with a projection
  # of the token embedding, giving each block its own input
  defp per_layer_inputs(input_ids, input_embeddings, embeddings, spec, opts) do
    name = opts[:name]
    size = spec.per_layer_input_size

    projection =
      embeddings
      |> Axon.dense(spec.num_blocks * size,
        kernel_initializer: kernel_initializer(spec),
        use_bias: false,
        name: join(name, "projection")
      )
      |> scale(:math.pow(spec.hidden_size, -0.5))
      |> split_blocks(spec)
      |> norm(spec, join(name, "projection_norm"))

    Layers.if_present input_embeddings do
      projection
    else
      input_ids
      |> Axon.embedding(spec.per_layer_input_vocab_size, spec.num_blocks * size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
      |> scale(:math.sqrt(size))
      |> split_blocks(spec)
      |> Axon.add(projection)
      |> scale(:math.pow(2, -0.5))
    end
  end

  defp split_blocks(input, spec) do
    Axon.nx(input, fn x ->
      {batch_size, sequence_length, _} = Nx.shape(x)
      Nx.reshape(x, {batch_size, sequence_length, spec.num_blocks, spec.per_layer_input_size})
    end)
  end

  defp decoder(
         hidden_state,
         per_layer_inputs,
         position_ids,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]
    layer_types = layer_types(spec)
    first_shared = first_shared_block(spec)

    producers =
      if first_shared < spec.num_blocks, do: producer_blocks(layer_types, first_shared), else: %{}

    Layers.Transformer.blocks(
      hidden_state,
      [
        num_blocks: spec.num_blocks,
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        cache: cache,
        name: join(name, "blocks")
      ],
      fn hidden_state, block, shared_key_values ->
        layer_type = Enum.at(layer_types, block.index)

        # Shared blocks reuse the key and value states of the last earlier
        # block of the same type, instead of computing their own
        shared_key_value =
          if block.index >= first_shared do
            Map.fetch!(shared_key_values, layer_type)
          end

        name = block.name
        shortcut = hidden_state

        {hidden_state, attention, key, value, self_attention_cache} =
          hidden_state
          |> norm(spec, join(name, "self_attention_norm"))
          |> self_attention(position_ids, block, layer_type, shared_key_value, spec,
            name: join(name, "self_attention")
          )

        hidden_state =
          hidden_state
          |> norm(spec, join(name, "post_attention_norm"))
          |> Axon.add(shortcut)

        intermediate_size =
          if spec.use_double_wide_ffn and shared_key_value != nil,
            do: spec.intermediate_size * 2,
            else: spec.intermediate_size

        shortcut = hidden_state

        hidden_state =
          hidden_state
          |> norm(spec, join(name, "pre_ffn_norm"))
          |> Layers.Transformer.gated_ffn(intermediate_size, spec.hidden_size,
            activation: spec.activation,
            kernel_initializer: kernel_initializer(spec),
            name: join(name, "ffn")
          )
          |> norm(spec, join(name, "post_ffn_norm"))
          |> Axon.add(shortcut)

        hidden_state =
          if per_layer_inputs do
            index = block.index
            per_layer_input = Axon.nx(per_layer_inputs, & &1[[.., .., index, ..]])
            shortcut = hidden_state

            hidden_state
            |> Axon.dense(spec.per_layer_input_size,
              kernel_initializer: kernel_initializer(spec),
              use_bias: false,
              name: join(name, "per_layer_input_gate")
            )
            |> Layers.activation(spec.activation)
            |> Axon.multiply(per_layer_input)
            |> Axon.dense(spec.hidden_size,
              kernel_initializer: kernel_initializer(spec),
              use_bias: false,
              name: join(name, "per_layer_input_projection")
            )
            |> norm(spec, join(name, "per_layer_input_norm"))
            |> Axon.add(shortcut)
          else
            hidden_state
          end

        hidden_state = layer_scale(hidden_state, name: join(name, "layer_scale"))

        result = %{
          hidden_state: hidden_state,
          attention: attention,
          self_attention_cache: self_attention_cache
        }

        shared_key_values =
          if block.index == producers[layer_type] do
            Map.put(shared_key_values, layer_type, {key, value})
          else
            shared_key_values
          end

        {result, shared_key_values}
      end,
      %{}
    )
    |> elem(0)
  end

  defp self_attention(hidden_state, position_ids, block, layer_type, shared_key_value, spec, opts) do
    name = opts[:name]
    full? = layer_type == :full_attention
    {num_key_value_heads, head_size} = key_value_shape(spec, layer_type)

    rotary_opts =
      if full? do
        [
          base: spec.rotary_embedding_base,
          percentage: spec.rotary_embedding_percentage,
          scaling_strategy: spec.rotary_embedding_scaling_strategy
        ]
      else
        [
          base: spec.rotary_embedding_base_local,
          percentage: spec.rotary_embedding_percentage_local,
          scaling_strategy: spec.rotary_embedding_scaling_strategy_local
        ]
      end

    rotary_opts =
      [position_ids: position_ids, max_positions: spec.max_positions] ++ rotary_opts

    rotary_layer_opts = [name: join(name, "rotary_embedding")]

    project = fn input, num_heads, suffix ->
      Layers.Transformer.project_heads(input, num_heads, head_size,
        use_bias: spec.use_attention_bias,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, suffix)
      )
    end

    query =
      hidden_state
      |> project.(spec.num_attention_heads, "query")
      |> norm(spec, join(name, "query_norm"))

    {query, key, value, cache} =
      case shared_key_value do
        {key, value} ->
          query =
            Layers.Transformer.rotary_embedding(
              query,
              block.attention_mask,
              head_size,
              rotary_opts,
              rotary_layer_opts
            )

          {query, key, value, block.self_attention_cache}

        nil ->
          key = project.(hidden_state, num_key_value_heads, "key")

          value =
            if full? and spec.attention_key_equals_value,
              do: key,
              else: project.(hidden_state, num_key_value_heads, "value")

          key = norm(key, spec, join(name, "key_norm"))

          {query, key} =
            Layers.Transformer.rotary_embedding(
              query,
              key,
              block.attention_mask,
              head_size,
              rotary_opts,
              rotary_layer_opts
            )

          value = unscaled_norm(value, spec)

          {key, value, cache} =
            Layers.Decoder.cached_attention_key_values(
              key,
              value,
              block.self_attention_cache,
              block.offset
            )

          {query, key, value, cache}
      end

    num_key_value_groups = div(spec.num_attention_heads, num_key_value_heads)

    attention_window_size =
      if full?, do: nil, else: {spec.attention_window_size - 1, spec.attention_window_size - 1}

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        Layers.Transformer.repeat_heads(key, num_key_value_groups),
        Layers.Transformer.repeat_heads(value, num_key_value_groups),
        block.attention_mask,
        block.attention_head_mask,
        Layers.none(),
        block.offset,
        scale: 1.0,
        causal: true,
        window_size: attention_window_size
      )

    attention_output =
      Layers.Transformer.output_projection(attention_output, spec.hidden_size,
        use_bias: spec.use_attention_bias,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output")
      )

    {attention_output, attention_weights, key, value, cache}
  end

  defp key_value_shape(spec, :full_attention) do
    num_key_value_heads =
      if spec.attention_key_equals_value,
        do: spec.global_num_key_value_heads || spec.num_key_value_heads,
        else: spec.num_key_value_heads

    {num_key_value_heads, spec.global_attention_head_size}
  end

  defp key_value_shape(spec, :sliding_attention) do
    {spec.num_key_value_heads, spec.attention_head_size}
  end

  defp norm(input, spec, name) do
    Layers.rms_norm(input, epsilon: spec.layer_norm_epsilon, upcast: :all, name: name)
  end

  defp unscaled_norm(input, spec) do
    Axon.nx(input, fn x ->
      y = Nx.as_type(x, :f32)
      variance = y |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)

      y
      |> Nx.multiply(Nx.rsqrt(Nx.add(variance, spec.layer_norm_epsilon)))
      |> Nx.as_type(Nx.type(x))
    end)
  end

  defp scale(input, factor) do
    Axon.nx(input, fn x -> Nx.multiply(x, Nx.tensor(factor, type: Nx.type(x))) end)
  end

  defp layer_scale(input, opts) do
    scalar = Axon.param("scalar", {1}, initializer: :ones)

    Axon.layer(
      fn x, scalar, _opts -> Nx.multiply(x, Nx.as_type(scalar, Nx.type(x))) end,
      [input, scalar],
      name: opts[:name],
      op_name: :layer_scale
    )
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    logits =
      Layers.dense_transposed(hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output")
      )

    case spec.final_logit_softcapping do
      nil ->
        logits

      cap ->
        Axon.nx(logits, fn x -> x |> Nx.divide(cap) |> Nx.tanh() |> Nx.multiply(cap) end)
    end
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp layer_types(spec) do
    spec.layer_types ||
      Enum.map(0..(spec.num_blocks - 1), fn index ->
        if rem(index + 1, 6) == 0, do: :full_attention, else: :sliding_attention
      end)
  end

  defp first_shared_block(spec), do: spec.num_blocks - spec.num_shared_key_value_blocks

  # The last block of each layer type before the shared ones produces the
  # key and value states that shared blocks of that type reuse
  defp producer_blocks(layer_types, first_shared) do
    layer_types
    |> Enum.take(first_shared)
    |> Enum.with_index()
    |> Map.new()
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, %{"model_type" => "gemma4", "text_config" => %{} = text_config} = data) do
      # The multimodal config controls whether the output head is tied.
      text_config =
        Map.put(text_config, "tie_word_embeddings", Map.get(data, "tie_word_embeddings", true))

      load(spec, text_config)
    end

    def load(spec, data) do
      import Shared.Converters

      if data["enable_moe_block"] do
        raise ArgumentError, "Gemma 4 mixture-of-experts models are not supported yet"
      end

      rotary_opts = Shared.rotary_embedding_options_from_transformers(data)

      # Standard RoPE in the checkpoint overrides the proportional default.
      rotary_opts =
        if get_in(data, ["rope_parameters", "full_attention"]) do
          Keyword.put_new(rotary_opts, :rotary_embedding_scaling_strategy, nil)
        else
          rotary_opts
        end

      data =
        data
        |> Map.put_new_lazy("global_head_dim", fn -> per_layer_value(data, "head_dim") end)
        |> Map.put_new_lazy("num_global_key_value_heads", fn ->
          per_layer_value(data, "num_key_value_heads")
        end)

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", number()},
          use_double_wide_ffn: {"use_double_wide_mlp", boolean()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          global_num_key_value_heads: {"num_global_key_value_heads", optional(number())},
          attention_key_equals_value: {"attention_k_eq_v", boolean()},
          attention_head_size: {"head_dim", number()},
          global_attention_head_size: {"global_head_dim", number()},
          activation: {"hidden_activation", activation()},
          use_attention_bias: {"attention_bias", boolean()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          attention_window_size: {"sliding_window", number()},
          layer_types:
            {"layer_types",
             list(
               mapping(%{
                 "sliding_attention" => :sliding_attention,
                 "full_attention" => :full_attention
               })
             )},
          per_layer_input_size: {"hidden_size_per_layer_input", number()},
          per_layer_input_vocab_size: {"vocab_size_per_layer_input", number()},
          num_shared_key_value_blocks: {"num_kv_shared_layers", number()},
          final_logit_softcapping: {"final_logit_softcapping", optional(number())},
          tie_word_embeddings: {"tie_word_embeddings", boolean()}
        ) ++ rotary_opts ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end

    # Newer transformers versions save the global attention shapes per layer, instead of
    # as "global_head_dim" and "num_global_key_value_heads"
    defp per_layer_value(data, key) do
      (data["per_layer_config"] || %{})
      |> Map.values()
      |> Enum.find_value(& &1[key])
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      # The loader infers the text model prefix from the checkpoint.
      %{
        "embedder.token_embedding" => "embed_tokens",
        "per_layer_inputs.token_embedding" => "embed_tokens_per_layer",
        "per_layer_inputs.projection" => "per_layer_model_projection",
        "per_layer_inputs.projection_norm" => "per_layer_projection_norm",
        "decoder.blocks.{n}.self_attention.query" => "layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention.query_norm" => "layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" => "layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.self_attention_norm" => "layers.{n}.input_layernorm",
        "decoder.blocks.{n}.post_attention_norm" => "layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.pre_ffn_norm" => "layers.{n}.pre_feedforward_layernorm",
        "decoder.blocks.{n}.post_ffn_norm" => "layers.{n}.post_feedforward_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.per_layer_input_gate" => "layers.{n}.per_layer_input_gate",
        "decoder.blocks.{n}.per_layer_input_projection" => "layers.{n}.per_layer_projection",
        "decoder.blocks.{n}.per_layer_input_norm" => "layers.{n}.post_per_layer_input_norm",
        "decoder.blocks.{n}.layer_scale" => %{
          "scalar" => {[{"layers.{n}", "layer_scalar"}], fn [value] -> value end}
        },
        "output_norm" => "norm",
        "language_modeling_head.output" => language_modeling_head_source(spec.tie_word_embeddings)
      }
    end

    defp language_modeling_head_source(true), do: "embed_tokens"

    defp language_modeling_head_source(false) do
      # Give the full parameter name so the loader can find it at the root.
      %{"kernel" => {[{"lm_head", "lm_head.weight"}], fn [weight] -> weight end}}
    end
  end
end
