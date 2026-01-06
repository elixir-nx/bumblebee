defmodule Bumblebee.Text.ModernBertDecoder do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50368,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 8192,
        doc: """
        the maximum sequence length that this model can process. ModernBERT Decoder uses RoPE
        (Rotary Position Embedding) instead of absolute position embeddings
        """
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 22,
        doc: "the number of Transformer blocks in the decoder"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      intermediate_size: [
        default: 1152,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the decoder"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function used in the gated FFN"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for embedding and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      local_attention_window: [
        default: 128,
        doc: "the window size for local attention layers"
      ],
      global_attention_every_n_layers: [
        default: 3,
        doc: "apply global attention every N layers (1 means every layer is global)"
      ],
      rotary_embedding_base_local: [
        default: 10_000.0,
        doc: "base for computing rotary embedding frequency for local (sliding) attention layers"
      ],
      rotary_embedding_base: [
        default: 160_000.0,
        doc: "base for computing rotary embedding frequency for global attention layers"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++ Shared.token_options(pad_token_id: 0)

  @moduledoc """
  ModernBERT Decoder model family.

  ModernBERT Decoder uses the same architecture as ModernBERT but is trained
  with a causal language modeling objective for text generation tasks.

  ## Architectures

    * `:base` - plain ModernBERT Decoder without any head on top

    * `:for_causal_language_modeling` - ModernBERT Decoder with a language modeling
      head. The head returns logits for each token in the original sequence

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

      Mask to nullify selected heads of the self-attention blocks in
      the decoder.

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

  ## References

    * [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)

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
    %{"input_ids" => Nx.template({1, 1}, :s64)}
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: div(spec.hidden_size, spec.num_attention_heads),
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
      layer_norm(decoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: "output_norm"
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

    Layers.default input_embeddings do
      input_ids
      |> Axon.embedding(spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
      |> layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
      |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
    end
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

    ffn_fun =
      &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
        activation: spec.activation,
        name: &2
      )

    rotary_embedding = [
      position_ids: position_ids,
      max_positions: spec.max_positions,
      base: spec.rotary_embedding_base
    ]

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cache: cache
    }

    for idx <- 0..(spec.num_blocks - 1), reduce: state do
      state ->
        block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])
        block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

        block_type =
          if idx == 0 do
            &block_without_self_attention_norm/3
          else
            :norm_first
          end

        {hidden_state, attention, _cross_attention, block_cache, _attention_relative_bias} =
          Layers.Transformer.block(state.hidden_state,
            attention_mask: attention_mask,
            attention_head_mask: block_attention_head_mask,
            block_cache: block_cache,
            num_attention_heads: spec.num_attention_heads,
            hidden_size: spec.hidden_size,
            kernel_initializer: kernel_initializer(spec),
            dropout_rate: spec.dropout_rate,
            attention_dropout_rate: spec.attention_dropout_rate,
            query_use_bias: false,
            key_use_bias: false,
            value_use_bias: false,
            output_use_bias: false,
            block_type: block_type,
            layer_norm: &layer_norm(&1, epsilon: spec.layer_norm_epsilon, name: &2),
            ffn: ffn_fun,
            causal: true,
            rotary_embedding: rotary_embedding,
            name: join(name, "blocks.#{idx}")
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention),
          cache: Layers.Decoder.put_block_cache(state.cache, idx, block_cache)
        }
    end
  end

  defp block_without_self_attention_norm(hidden_state, steps, _name) do
    shortcut = hidden_state
    {hidden_state, attention_info} = steps.self_attention.(hidden_state)
    hidden_state = Axon.add(hidden_state, shortcut)

    {hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn hidden_state ->
        shortcut = hidden_state

        {hidden_state, cross_attention_info} =
          hidden_state
          |> steps.cross_attention_norm.()
          |> steps.cross_attention.()

        {Axon.add(hidden_state, shortcut), cross_attention_info}
      end)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> steps.output_norm.()
      |> steps.ffn.()
      |> Axon.add(shortcut)

    {hidden_state, attention_info, cross_attention_info}
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        use_bias: false,
        name: join(name, "intermediate")
      )

    gate =
      Axon.dense(hidden_state, intermediate_size, use_bias: false, name: join(name, "gate"))

    hidden_state = Axon.multiply(Layers.activation(intermediate, activation), gate)

    Axon.dense(hidden_state, output_size, use_bias: false, name: join(name, "output"))
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.hidden_size,
      use_bias: false,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Layers.activation(spec.activation)
    |> layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Layers.dense_transposed(spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.bias(name: join(name, "bias"))
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp layer_norm(x, opts) do
    name = opts[:name]
    epsilon = opts[:epsilon] || 1.0e-5

    Axon.layer(
      fn x, gamma, _opts ->
        Axon.Layers.layer_norm(x, gamma, Nx.broadcast(0.0, gamma), epsilon: epsilon)
      end,
      [x, Axon.param("weight", fn shape -> {elem(shape, tuple_size(shape) - 1)} end)],
      name: name,
      op_name: :layer_norm
    )
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_activation", activation()},
          dropout_rate: {"embedding_dropout", optional(number())},
          attention_dropout_rate: {"attention_dropout", optional(number())},
          layer_norm_epsilon: {"norm_eps", optional(number())},
          initializer_scale: {"initializer_range", optional(number())},
          local_attention_window: {"sliding_window", number()},
          global_attention_every_n_layers: {"global_attn_every_n_layers", number()},
          rotary_embedding_base_local: {"local_rope_theta", optional(number())},
          rotary_embedding_base: {"global_rope_theta", optional(number())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "model.embeddings.tok_embeddings",
        "embedder.norm" => "model.embeddings.norm",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.attn.Wo",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.attn_norm",
        "decoder.blocks.{n}.ffn.intermediate" =>
          Shared.sliced_dense_params_source(
            "model.layers.{n}.mlp.Wi",
            {[1, 1], :auto},
            0
          ),
        "decoder.blocks.{n}.ffn.gate" =>
          Shared.sliced_dense_params_source(
            "model.layers.{n}.mlp.Wi",
            {[1, 1], :auto},
            1
          ),
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.Wo",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.mlp_norm",
        "output_norm" => "model.final_norm",
        "language_modeling_head.dense" => "lm_head.dense",
        "language_modeling_head.norm" => "lm_head.norm",
        "language_modeling_head.output" => "model.embeddings.tok_embeddings",
        "language_modeling_head.bias" => "decoder"
      }
    end
  end
end
