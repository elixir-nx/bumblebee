defmodule Bumblebee.Text.NemotronH do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 131_072,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 4096,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 4096,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 21_504,
        doc: "the dimensionality of intermediate layers"
      ],
      num_blocks: [
        default: 4,
        doc: "the number of blocks in the model"
      ],
      block_types: [
        default: [:mamba, :moe, :attention, :mlp],
        doc: """
        a list with the type of each block, one of `:mamba` (a state-space mixer), `:attention`,
        `:mlp` or `:moe`
        """
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention block in the model"
      ],
      num_key_value_heads: [
        default: 8,
        doc: "the number of key value heads for each attention block in the model"
      ],
      attention_head_size: [
        default: 128,
        doc: "the size of the key, value, and query projection per attention head"
      ],
      mamba_num_heads: [
        default: 128,
        doc: "the number of heads in each state-space block"
      ],
      mamba_head_size: [
        default: 64,
        doc: "the dimensionality of each state-space head"
      ],
      mamba_state_size: [
        default: 128,
        doc: "the size of the recurrent state per state-space head channel"
      ],
      mamba_num_groups: [
        default: 8,
        doc: """
        the number of groups that the state-space heads are split into. Heads within a group
        share the input-dependent state transformations
        """
      ],
      mamba_activation: [
        default: :silu,
        doc: "the activation applied after the convolution in the state-space blocks"
      ],
      conv_kernel_size: [
        default: 4,
        doc: "the size of the short convolution in the state-space blocks"
      ],
      chunk_size: [
        default: 128,
        doc: "the number of positions processed in a single chunk of the state-space recurrence"
      ],
      time_step_min: [
        default: 0.001,
        doc: "the lower bound that the state-space time step is clamped to"
      ],
      activation: [
        default: :relu_squared,
        doc: "the activation function in the feed-forward networks"
      ],
      moe_intermediate_size: [
        default: 7688,
        doc: "the dimensionality of intermediate layers in each mixture-of-experts expert"
      ],
      moe_shared_expert_intermediate_size: [
        default: 7688,
        doc: "the dimensionality of intermediate layers in the shared expert"
      ],
      moe_latent_size: [
        default: nil,
        doc: "when set, the mixture-of-experts input and output are projected to a latent size"
      ],
      num_experts: [
        default: 8,
        doc: "the number of experts in each mixture-of-experts block"
      ],
      num_experts_per_token: [
        default: 2,
        doc: "the number of experts that each token is routed to"
      ],
      num_expert_groups: [
        default: 1,
        doc: "the number of groups that the experts are split into for group-limited routing"
      ],
      num_expert_groups_per_token: [
        default: 1,
        doc: "the number of expert groups that each token is routed to"
      ],
      normalize_top_k_probabilities: [
        default: true,
        doc: "whether to normalize the weights of the selected experts to sum up to one"
      ],
      routed_scaling_factor: [
        default: 1.0,
        doc: "the constant that the routing weights are multiplied by"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
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
      use_bias: [
        default: false,
        doc: "whether to use bias in the state-space projections and feed-forward networks"
      ],
      use_conv_bias: [
        default: true,
        doc: "whether to use bias in the state-space convolution"
      ]
    ] ++ Shared.token_options(pad_token_id: 0)

  @moduledoc """
  Nemotron-H model family.

  This is a hybrid model, where each block has a single mixer, which is
  either a Mamba-2 state-space mixer, self-attention, a feed-forward
  network or a mixture-of-experts network. Most of the blocks are
  state-space mixers, so the memory does not grow with the sequence
  length.

  ## Architectures

    * `:base` - plain Nemotron-H without any head on top

    * `:for_causal_language_modeling` - Nemotron-H with a language
      modeling head. The head returns logits for each token in the
      original sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

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

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers
  alias Bumblebee.Layers.Mamba
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
    block_types = spec.block_types

    attention_block? = fn idx -> Enum.at(block_types, idx) == :attention end
    mamba_block? = fn idx -> Enum.at(block_types, idx) == :mamba end

    conv_size =
      spec.mamba_num_heads * spec.mamba_head_size +
        2 * spec.mamba_num_groups * spec.mamba_state_size

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      # Only the attention blocks have a key-value cache, for the
      # remaining ones we keep a placeholder of minimal size
      decoder_num_attention_heads: fn idx ->
        if attention_block?.(idx), do: spec.num_attention_heads, else: 1
      end,
      attention_head_size: fn idx ->
        if attention_block?.(idx), do: spec.attention_head_size, else: 1
      end,
      decoder_num_blocks: spec.num_blocks,
      extra_window_states: fn idx ->
        if mamba_block?.(idx) do
          [
            convolution: {spec.conv_kernel_size - 1, conv_size},
            state: {spec.mamba_num_heads, spec.mamba_head_size, spec.mamba_state_size}
          ]
        else
          []
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

    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
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

    decoder_outputs =
      decoder(
        embeddings,
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

  defp decoder(hidden_state, attention_mask, attention_head_mask, cache, spec, opts) do
    name = opts[:name]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cache: cache
    }

    outputs =
      for idx <- 0..(spec.num_blocks - 1), reduce: state do
        state ->
          block_name = join(join(name, "blocks"), idx)

          block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

          {attention_cache, cross_attention_cache} =
            Layers.Decoder.get_attention_caches(block_cache)

          shortcut = state.hidden_state

          hidden_state =
            Layers.rms_norm(state.hidden_state,
              name: join(block_name, "norm"),
              epsilon: spec.layer_norm_epsilon
            )

          {hidden_state, attention, attention_cache} =
            case Enum.at(spec.block_types, idx) do
              :mamba ->
                {hidden_state, attention_cache} =
                  Mamba.mixer(hidden_state, attention_mask, attention_cache, offset,
                    hidden_size: spec.hidden_size,
                    num_heads: spec.mamba_num_heads,
                    head_size: spec.mamba_head_size,
                    state_size: spec.mamba_state_size,
                    num_groups: spec.mamba_num_groups,
                    conv_kernel_size: spec.conv_kernel_size,
                    chunk_size: spec.chunk_size,
                    activation: spec.mamba_activation,
                    time_step_min: spec.time_step_min,
                    layer_norm_epsilon: spec.layer_norm_epsilon,
                    use_bias: spec.use_bias,
                    use_conv_bias: spec.use_conv_bias,
                    kernel_initializer: kernel_initializer(spec),
                    name: join(block_name, "mixer")
                  )

                {hidden_state, Layers.none(), attention_cache}

              :attention ->
                block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])

                {hidden_state, attention, attention_cache, _} =
                  Layers.Transformer.multi_head_attention(
                    hidden_state,
                    hidden_state,
                    hidden_state,
                    attention_mask: attention_mask,
                    attention_head_mask: block_attention_head_mask,
                    attention_cache: attention_cache,
                    offset: offset,
                    causal: true,
                    num_heads: spec.num_attention_heads,
                    num_key_value_heads: spec.num_key_value_heads,
                    hidden_size: spec.hidden_size,
                    attention_head_size: spec.attention_head_size,
                    kernel_initializer: kernel_initializer(spec),
                    query_use_bias: false,
                    key_use_bias: false,
                    value_use_bias: false,
                    output_use_bias: false,
                    name: join(block_name, "mixer")
                  )

                {hidden_state, attention, attention_cache}

              :mlp ->
                hidden_state =
                  Moe.ffn(hidden_state, spec.intermediate_size, spec.hidden_size,
                    activation: spec.activation,
                    use_bias: spec.use_bias,
                    kernel_initializer: kernel_initializer(spec),
                    name: join(block_name, "mixer")
                  )

                {hidden_state, Layers.none(), attention_cache}

              :moe ->
                {moe_block(hidden_state, spec, join(block_name, "mixer")), Layers.none(),
                 attention_cache}
            end

          hidden_state = Axon.add(hidden_state, shortcut)

          block_cache =
            Layers.Decoder.put_attention_caches(
              block_cache,
              attention_cache,
              cross_attention_cache
            )

          %{
            hidden_state: hidden_state,
            hidden_states: Layers.append(state.hidden_states, hidden_state),
            attentions: Layers.append(state.attentions, attention),
            cache: Layers.Decoder.put_block_cache(state.cache, idx, block_cache)
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  defp moe_block(hidden_state, spec, name) do
    if spec.moe_latent_size do
      raise ArgumentError,
            "Nemotron-H with a mixture-of-experts latent projection is not supported"
    end

    Moe.block(hidden_state,
      num_experts: spec.num_experts,
      num_experts_per_token: spec.num_experts_per_token,
      hidden_size: spec.hidden_size,
      intermediate_size: spec.moe_intermediate_size,
      shared_expert_intermediate_size: spec.moe_shared_expert_intermediate_size,
      expert_group:
        if(spec.num_expert_groups > 1,
          do: [
            num_groups: spec.num_expert_groups,
            num_groups_per_token: spec.num_expert_groups_per_token
          ]
        ),
      activation: spec.activation,
      scoring: :sigmoid,
      normalize_top_k: spec.normalize_top_k_probabilities,
      routed_scaling_factor: spec.routed_scaling_factor,
      expert_variant: :ungated,
      kernel_initializer: kernel_initializer(spec),
      name: name
    )
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
          activation: {"mlp_hidden_act", activation()},
          mamba_activation: {"mamba_hidden_act", activation()},
          mamba_num_heads: {"mamba_num_heads", number()},
          mamba_head_size: {"mamba_head_dim", number()},
          mamba_state_size: {"ssm_state_size", number()},
          mamba_num_groups: {"n_groups", number()},
          conv_kernel_size: {"conv_kernel", number()},
          chunk_size: {"chunk_size", number()},
          time_step_min: {"time_step_min", number()},
          use_bias: {"use_bias", boolean()},
          use_conv_bias: {"use_conv_bias", boolean()},
          moe_intermediate_size: {"moe_intermediate_size", number()},
          moe_shared_expert_intermediate_size: {"moe_shared_expert_intermediate_size", number()},
          moe_latent_size: {"moe_latent_size", optional(number())},
          num_experts: {"n_routed_experts", number()},
          num_experts_per_token: {"num_experts_per_tok", number()},
          num_expert_groups: {"n_group", number()},
          num_expert_groups_per_token: {"topk_group", number()},
          normalize_top_k_probabilities: {"norm_topk_prob", boolean()},
          routed_scaling_factor: {"routed_scaling_factor", number()},
          layer_norm_epsilon: {"layer_norm_epsilon", number()},
          initializer_scale: {"initializer_range", number()},
          block_types:
            {"layers_block_type",
             list(
               mapping(%{
                 "linear_attention" => :mamba,
                 "full_attention" => :attention,
                 "mlp" => :mlp,
                 "moe" => :moe
               })
             )}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.Layers.Moe

    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => %{
          "kernel" => {
            [
              [
                {"backbone.embeddings", "weight"},
                {"backbone.embedding", "weight"}
              ]
            ],
            fn [embeddings] -> embeddings end
          }
        },
        "decoder.blocks.{n}.norm" => "backbone.layers.{n}.norm",
        # Attention blocks
        "decoder.blocks.{n}.mixer.query" => "backbone.layers.{n}.mixer.q_proj",
        "decoder.blocks.{n}.mixer.key" => "backbone.layers.{n}.mixer.k_proj",
        "decoder.blocks.{n}.mixer.value" => "backbone.layers.{n}.mixer.v_proj",
        # State-space blocks
        "decoder.blocks.{n}.mixer.input" => "backbone.layers.{n}.mixer.in_proj",
        "decoder.blocks.{n}.mixer.convolution" => %{
          "kernel" => {
            [{"backbone.layers.{n}.mixer.conv1d", "weight"}],
            fn [kernel] -> Nx.squeeze(kernel, axes: [1]) end
          },
          "bias" => {
            [{"backbone.layers.{n}.mixer.conv1d", "bias"}],
            fn [bias] -> bias end
          }
        },
        "decoder.blocks.{n}.mixer.a_log" => parameter("backbone.layers.{n}.mixer", "A_log"),
        "decoder.blocks.{n}.mixer.d" => parameter("backbone.layers.{n}.mixer", "D"),
        "decoder.blocks.{n}.mixer.time_step_bias" =>
          parameter("backbone.layers.{n}.mixer", "dt_bias"),
        "decoder.blocks.{n}.mixer.output_norm" => "backbone.layers.{n}.mixer.norm",
        # Both the attention and the state-space blocks have an output
        # projection, they are just named differently
        "decoder.blocks.{n}.mixer.output" => %{
          "kernel" => {
            [
              [
                {"backbone.layers.{n}.mixer.out_proj", "weight"},
                {"backbone.layers.{n}.mixer.o_proj", "weight"},
                {"backbone.layers.{n}.mixer.down_proj", "weight"}
              ]
            ],
            fn [kernel] -> Nx.transpose(kernel) end
          }
        },
        # Feed-forward blocks
        "decoder.blocks.{n}.mixer.intermediate" => "backbone.layers.{n}.mixer.up_proj",
        # Mixture-of-experts blocks
        "decoder.blocks.{n}.mixer.router" => %{
          "kernel" => {
            [{"backbone.layers.{n}.mixer.gate", "weight"}],
            fn [kernel] -> Nx.transpose(kernel) end
          },
          "score_correction_bias" => {
            [{"backbone.layers.{n}.mixer.gate", "e_score_correction_bias"}],
            fn [bias] -> bias end
          }
        },
        "decoder.blocks.{n}.mixer.shared_expert.intermediate" =>
          "backbone.layers.{n}.mixer.shared_experts.up_proj",
        "decoder.blocks.{n}.mixer.shared_expert.output" =>
          "backbone.layers.{n}.mixer.shared_experts.down_proj",
        "output_norm" => "backbone.norm_f",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "backbone.embeddings", else: "lm_head")
      }
      |> Map.merge(expert_params(spec))
    end

    defp parameter(source, name) do
      %{"value" => {[{source, name}], fn [value] -> value end}}
    end

    defp expert_params(spec) do
      experts = "backbone.layers.{n}.mixer.experts"

      refs = fn projection ->
        for idx <- 0..(spec.num_experts - 1) do
          [
            {experts <> ".#{idx}.#{projection}", "weight"},
            {experts, projection}
          ]
        end
      end

      %{
        "decoder.blocks.{n}.mixer.experts" => %{
          "up_kernel" => {refs.("up_proj"), &Moe.stack_expert_kernels(&1)},
          "down_kernel" => {refs.("down_proj"), &Moe.stack_expert_kernels(&1)}
        }
      }
    end
  end
end
