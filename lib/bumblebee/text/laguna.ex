defmodule Bumblebee.Text.Laguna do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 100_352,
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
        default: 2048,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 8192,
        doc: "the dimensionality of intermediate layers"
      ],
      attention_head_size: [
        default: 128,
        doc: """
        the size of the key, value, and query projection per attention head.
        """
      ],
      num_blocks: [
        default: 40,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 48,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 8,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 500_000,
        doc: "base for computing rotary embedding frequency"
      ],
      rotary_embedding_scaling_strategy: [
        default: nil,
        doc: """
        scaling configuration for rotary embedding. Currently the supported values are:

          * `%{type: :linear, factor: number()}`

          * `%{type: :dynamic, factor: number()}`

        For more details see https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
        """
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
      ],
      block_types: [
        default: nil,
        doc: """
        a list with the attention type of each block, either `:full_attention` or
        `:sliding_attention`. When `nil`, every block uses full attention
        """
      ],
      block_num_attention_heads: [
        default: nil,
        doc: """
        a list with the number of attention heads of each block. When `nil`, every block uses
        `:num_attention_heads`
        """
      ],
      attention_window_size: [
        default: 512,
        doc: "the size of the attention window for blocks using sliding window attention"
      ],
      local_rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency in the sliding window blocks"
      ],
      rotary_embedding_percentage: [
        default: 0.5,
        doc: """
        the fraction of each head that rotary embedding is applied to in the full attention
        blocks
        """
      ],
      local_rotary_embedding_percentage: [
        default: 1.0,
        doc: """
        the fraction of each head that rotary embedding is applied to in the sliding window
        blocks
        """
      ],
      shared_expert_intermediate_size: [
        default: 512,
        doc: "the dimensionality of intermediate layers in the shared expert"
      ],
      routed_scaling_factor: [
        default: 1.0,
        doc: "the constant that the routing weights are multiplied by"
      ],
      router_logits_softcapping: [
        default: nil,
        doc: "the value used for soft capping the router logits"
      ],
      use_attention_bias: [
        default: false,
        doc: "whether to use bias in the query, key, value and output projections"
      ],
      moe_intermediate_size: [
        default: 512,
        doc: "the dimensionality of intermediate layers in each mixture-of-experts expert"
      ],
      num_experts: [
        default: 256,
        doc: "the number of experts in each mixture-of-experts block"
      ],
      num_experts_per_token: [
        default: 8,
        doc: "the number of experts that each token is routed to"
      ],
      normalize_top_k_probabilities: [
        default: true,
        doc: "whether to normalize the weights of the selected experts to sum up to one"
      ],
      block_mlp_types: [
        default: nil,
        doc: """
        a list with the feed-forward network type for each block, either `:dense` or `:sparse`.
        Defaults to all blocks using a mixture-of-experts network
        """
      ]
    ] ++
      Shared.token_options(pad_token_id: 151_643)

  @moduledoc """
  Laguna model family.

  The blocks alternate between sliding window and full attention, and
  the two kinds differ in the number of attention heads and in the
  rotary embedding. The attention output is gated, with one gate value
  per head, and most blocks use a mixture-of-experts network with both
  routed and shared experts.

  ## Architectures

    * `:base` - plain Laguna without any head on top

    * `:for_causal_language_modeling` - Laguna with a language modeling
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
  alias Bumblebee.Layers.Moe

  @impl true
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling
    ]

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
    block_num_attention_heads = block_num_attention_heads(spec)

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      decoder_num_attention_heads: &Enum.at(block_num_attention_heads, &1),
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
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
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
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
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

    block_types = block_types(spec)
    block_num_attention_heads = block_num_attention_heads(spec)

    query_norm =
      &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, channel_index: -1, name: &2)

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

    rotary_embedding = fn idx ->
      {base, percentage} =
        case Enum.at(block_types, idx) do
          :full_attention ->
            {spec.rotary_embedding_base, spec.rotary_embedding_percentage}

          :sliding_attention ->
            {spec.local_rotary_embedding_base, spec.local_rotary_embedding_percentage}
        end

      [
        position_ids: position_ids,
        max_positions: spec.max_positions,
        base: base,
        percentage: percentage,
        scaling_strategy:
          if(Enum.at(block_types, idx) == :full_attention,
            do: spec.rotary_embedding_scaling_strategy
          )
      ]
    end

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: &Enum.at(block_num_attention_heads, &1),
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      kernel_initializer: kernel_initializer(spec),
      query_use_bias: spec.use_attention_bias,
      key_use_bias: spec.use_attention_bias,
      value_use_bias: spec.use_attention_bias,
      output_use_bias: spec.use_attention_bias,
      block_type: :norm_first,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      causal: true,
      layer_norm: &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, name: &2),
      ffn: &block_ffn(&1, spec),
      rotary_embedding: rotary_embedding,
      attention_window_size: attention_window_size,
      query_norm: query_norm,
      key_norm: query_norm,
      # The attention output is gated, with a single value per head
      output_gate_activation: :softplus,
      output_gate_per_head: true,
      name: join(name, "blocks")
    )
  end

  defp block_ffn(block_idx, spec) do
    if sparse_block?(spec, block_idx) do
      &Moe.block(&1,
        num_experts: spec.num_experts,
        num_experts_per_token: spec.num_experts_per_token,
        hidden_size: spec.hidden_size,
        intermediate_size: spec.moe_intermediate_size,
        shared_expert_intermediate_size: spec.shared_expert_intermediate_size,
        activation: spec.activation,
        scoring: :sigmoid,
        normalize_top_k: spec.normalize_top_k_probabilities,
        routed_scaling_factor: spec.routed_scaling_factor,
        logit_softcapping: spec.router_logits_softcapping,
        kernel_initializer: kernel_initializer(spec),
        name: &2
      )
    else
      &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
        name: &2,
        activation: spec.activation
      )
    end
  end

  defp sparse_block?(spec, block_idx) do
    spec.num_experts > 0 and Enum.at(block_mlp_types(spec), block_idx) == :sparse
  end

  defp block_mlp_types(spec) do
    spec.block_mlp_types || List.duplicate(:sparse, spec.num_blocks)
  end

  defp block_types(spec) do
    spec.block_types || List.duplicate(:full_attention, spec.num_blocks)
  end

  defp block_num_attention_heads(spec) do
    spec.block_num_attention_heads || List.duplicate(spec.num_attention_heads, spec.num_blocks)
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

    hidden_state = Axon.multiply(intermediate, Axon.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
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

      data = Shared.normalize_rope_options(data)

      scaling_strategy_converter = fn _name, value ->
        case value do
          %{"type" => "linear", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :linear, factor: factor}}

          %{"type" => "dynamic", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :dynamic, factor: factor}}

          nil ->
            {:ok, nil}

          _other ->
            {:ok, nil}
        end
      end

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          rotary_embedding_base: {"rope_theta", number()},
          rotary_embedding_scaling_strategy:
            {"rope_scaling", optional(scaling_strategy_converter)},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          moe_intermediate_size: {"moe_intermediate_size", number()},
          shared_expert_intermediate_size: {"shared_expert_intermediate_size", number()},
          routed_scaling_factor: {"moe_routed_scaling_factor", number()},
          router_logits_softcapping: {"moe_router_logit_softcapping", optional(number())},
          attention_window_size: {"sliding_window", number()},
          use_attention_bias: {"attention_bias", boolean()},
          block_num_attention_heads: {"num_attention_heads_per_layer", optional(list(number()))},
          block_types:
            {"layer_types",
             optional(
               list(
                 mapping(%{
                   "full_attention" => :full_attention,
                   "sliding_attention" => :sliding_attention
                 })
               )
             )},
          num_experts: {"num_local_experts", number()},
          num_experts_per_token: {"num_experts_per_tok", number()},
          normalize_top_k_probabilities: {"norm_topk_prob", boolean()},
          block_mlp_types:
            {"mlp_layer_types",
             optional(list(mapping(%{"dense" => :dense, "sparse" => :sparse})))}
        ) ++ Shared.common_options_from_transformers(data, spec)

      opts = Keyword.merge(opts, rotary_options(data))

      # A softcapping of zero means no softcapping
      opts =
        case opts[:router_logits_softcapping] do
          +0.0 -> Keyword.put(opts, :router_logits_softcapping, nil)
          _other -> opts
        end

      # Older checkpoints name the number of experts differently
      opts =
        case data["num_experts"] do
          nil -> opts
          num_experts -> Keyword.put_new(opts, :num_experts, num_experts)
        end

      @for.config(spec, opts)
    end

    defp rotary_options(data) do
      case data["rope_parameters"] do
        %{"full_attention" => %{} = full, "sliding_attention" => %{} = local} ->
          import Shared.Converters

          scaling =
            case rope_scaling_strategy().("rope_parameters", full) do
              {:ok, strategy} -> strategy
              {:error, _} -> nil
            end

          [
            rotary_embedding_base: full["rope_theta"],
            rotary_embedding_percentage: full["partial_rotary_factor"] || 1.0,
            rotary_embedding_scaling_strategy: scaling,
            local_rotary_embedding_base: local["rope_theta"],
            local_rotary_embedding_percentage: local["partial_rotary_factor"] || 1.0
          ]

        _other ->
          []
      end
    end
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
        "decoder.blocks.{n}.self_attention.output_gate" => "model.layers.{n}.self_attn.g_proj",
        "decoder.blocks.{n}.ffn.shared_expert.gate" =>
          "model.layers.{n}.mlp.shared_expert.gate_proj",
        "decoder.blocks.{n}.ffn.shared_expert.intermediate" =>
          "model.layers.{n}.mlp.shared_expert.up_proj",
        "decoder.blocks.{n}.ffn.shared_expert.output" =>
          "model.layers.{n}.mlp.shared_expert.down_proj",
        "decoder.blocks.{n}.self_attention.query_norm" => "model.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" => "model.layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.ffn.router" => %{
          "kernel" => {
            [{"model.layers.{n}.mlp.gate", "weight"}],
            fn [kernel] -> Nx.transpose(kernel) end
          },
          "score_correction_bias" => {
            [{"model.layers.{n}.mlp.experts", "e_score_correction_bias"}],
            fn [bias] -> bias end
          }
        },
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.post_attention_layernorm",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head")
      }
      |> Map.merge(expert_params(spec))
    end

    defp expert_params(spec) do
      experts = "model.layers.{n}.mlp.experts"

      refs = fn projection, packed ->
        for idx <- 0..(spec.num_experts - 1) do
          [
            {experts <> ".#{idx}.#{projection}", "weight"},
            {experts, packed}
          ]
        end
      end

      %{
        "decoder.blocks.{n}.ffn.experts" => %{
          "gate_kernel" =>
            {refs.("gate_proj", "gate_up_proj"),
             &Moe.stack_expert_kernels(&1, 0, spec.moe_intermediate_size)},
          "up_kernel" =>
            {refs.("up_proj", "gate_up_proj"),
             &Moe.stack_expert_kernels(
               &1,
               spec.moe_intermediate_size,
               spec.moe_intermediate_size
             )},
          "down_kernel" => {refs.("down_proj", "down_proj"), &Moe.stack_expert_kernels(&1)}
        }
      }
    end
  end
end
