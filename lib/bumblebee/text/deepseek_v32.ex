defmodule Bumblebee.Text.DeepseekV32 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 129_280,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 163_840,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 7168,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 18_432,
        doc: "the dimensionality of intermediate layers in the dense feed-forward blocks"
      ],
      moe_intermediate_size: [
        default: 2048,
        doc: "the dimensionality of intermediate layers in each mixture-of-experts expert"
      ],
      num_blocks: [
        default: 61,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 128,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      query_lora_rank: [
        default: 1536,
        doc: """
        the rank of the low-rank projection for queries. When `nil`, queries are projected
        directly from the hidden state
        """
      ],
      key_value_lora_rank: [
        default: 512,
        doc: "the rank of the low-rank projection for keys and values (the compressed latent)"
      ],
      qk_nope_head_size: [
        default: 128,
        doc: "the size of the non-positional part of query and key projections per attention head"
      ],
      qk_rope_head_size: [
        default: 64,
        doc: "the size of the rotary part of query and key projections per attention head"
      ],
      value_head_size: [
        default: 128,
        doc: "the size of the value projection per attention head"
      ],
      num_experts: [
        default: 256,
        doc: "the number of routed experts in each mixture-of-experts block"
      ],
      num_experts_per_token: [
        default: 8,
        doc: "the number of experts that each token is routed to"
      ],
      num_shared_experts: [
        default: 1,
        doc: "the number of shared experts, which are applied to all tokens"
      ],
      num_expert_groups: [
        default: 8,
        doc: "the number of groups the routed experts are split into"
      ],
      num_expert_groups_per_token: [
        default: 4,
        doc: "the number of expert groups that each token is routed to"
      ],
      first_dense_blocks: [
        default: 3,
        doc: """
        the number of leading Transformer blocks that use a regular dense feed-forward network,
        the remaining blocks use a mixture-of-experts block
        """
      ],
      routed_scaling_factor: [
        default: 2.5,
        doc: "the constant that the routing weights are multiplied by"
      ],
      normalize_top_k_probabilities: [
        default: true,
        doc: "whether to normalize the weights of the selected experts to sum up to one"
      ],
      activation: [
        default: :silu,
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

        For more details see https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
        """
      ],
      rotary_embedding_interleaved: [
        default: true,
        doc: """
        whether the rotary dimensions are laid out as interleaved pairs, rather than as two
        halves
        """
      ],
      index_top_k: [
        default: 2048,
        doc: """
        the number of keys that the sparse attention indexer selects for each query. When the
        key sequence is not longer than this number, the attention is effectively dense
        """
      ],
      index_num_heads: [
        default: 64,
        doc: "the number of scoring heads in the sparse attention indexer"
      ],
      index_head_size: [
        default: 128,
        doc: "the size of query and key projections per head in the sparse attention indexer"
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
        default: false,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 0)

  @moduledoc """
  DeepSeek V3.2 model family.

  The architecture builds on DeepSeek V3, that is, Multi-head Latent
  Attention (MLA) combined with a mixture-of-experts feed-forward network.
  On top of that, each block uses DeepSeek Sparse Attention (DSA), where
  a lightweight indexer scores all keys against the query and only the
  highest scoring `:index_top_k` keys are attended to.

  ## Architectures

    * `:base` - plain DeepSeek V3.2 without any head on top

    * `:for_causal_language_modeling` - DeepSeek V3.2 with a language
      modeling head. The head returns logits for each token in the
      original sequence

    * `:for_sequence_classification` - DeepSeek V3.2 with a sequence
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

  import Nx.Defn
  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers
  alias Bumblebee.Layers.Moe

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
      attention_head_size: qk_head_size(spec),
      key_head_size: qk_head_size(spec),
      value_head_size: spec.value_head_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks,
      extra_states: [indexer_key: {spec.index_head_size}]
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

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      causal: true,
      block_type: :norm_first,
      layer_norm: &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, name: &2),
      attention: &latent_attention(&1, &2, spec, position_ids),
      ffn: &block_ffn(&1, spec),
      name: join(name, "blocks")
    )
  end

  # Multi-head Latent Attention (MLA). Queries and keys/values are
  # projected through a low-rank latent space. Additionally, each key/query
  # is split into a non-positional part and a rotary part, where the rotary
  # part of the key is shared across all heads.
  defp latent_attention(hidden_state, opts, spec, position_ids) do
    name = opts[:name]

    num_heads = spec.num_attention_heads
    qk_head_size = qk_head_size(spec)

    query_latent =
      hidden_state
      |> Axon.dense(spec.query_lora_rank,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query_down"),
        use_bias: spec.use_attention_bias
      )
      |> Layers.rms_norm(
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "query_norm")
      )

    query =
      query_latent
      |> Axon.dense(num_heads * qk_head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query_up"),
        use_bias: false
      )
      |> Layers.split_heads(num_heads)

    key_value =
      Axon.dense(hidden_state, spec.key_value_lora_rank + spec.qk_rope_head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key_value_down"),
        use_bias: spec.use_attention_bias
      )

    key_value_latent =
      key_value
      |> Axon.nx(& &1[[.., .., 0..(spec.key_value_lora_rank - 1)//1]])
      |> Layers.rms_norm(
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "key_value_norm")
      )

    # The rotary part of the key is shared across heads
    key_rotary =
      key_value
      |> Axon.nx(& &1[[.., .., spec.key_value_lora_rank..-1//1]])
      |> Axon.nx(&Nx.new_axis(&1, 2))

    query_pass = Axon.nx(query, & &1[[.., .., .., 0..(spec.qk_nope_head_size - 1)//1]])
    query_rotary = Axon.nx(query, & &1[[.., .., .., spec.qk_nope_head_size..-1//1]])

    {query_rotary, key_rotary} =
      Layers.rotary_embedding(
        query_rotary,
        key_rotary,
        position_ids,
        opts[:attention_mask],
        spec.qk_rope_head_size,
        name: join(name, "rotary_embedding"),
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base,
        scaling_strategy: spec.rotary_embedding_scaling_strategy,
        interleaved: spec.rotary_embedding_interleaved
      )

    query = Axon.concatenate([query_pass, query_rotary], axis: -1)

    key_value_up =
      key_value_latent
      |> Axon.dense(num_heads * (spec.qk_nope_head_size + spec.value_head_size),
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key_value_up"),
        use_bias: false
      )
      |> Layers.split_heads(num_heads)

    key_pass = Axon.nx(key_value_up, & &1[[.., .., .., 0..(spec.qk_nope_head_size - 1)//1]])
    value = Axon.nx(key_value_up, & &1[[.., .., .., spec.qk_nope_head_size..-1//1]])

    key =
      Axon.layer(
        fn key_pass, key_rotary, _opts ->
          key_rotary = Nx.broadcast(key_rotary, put_elem(Nx.shape(key_rotary), 2, num_heads))
          Nx.concatenate([key_pass, key_rotary], axis: -1)
        end,
        [key_pass, key_rotary]
      )

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(
        key,
        value,
        opts[:attention_cache],
        opts[:offset]
      )

    {sparse_attention_bias, attention_cache} =
      indexer(
        hidden_state,
        query_latent,
        position_ids,
        opts[:attention_mask],
        attention_cache,
        opts[:offset],
        spec,
        name: join(name, "indexer")
      )

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        opts[:attention_mask],
        opts[:attention_head_mask],
        sparse_attention_bias,
        opts[:offset],
        scale: attention_scale(spec),
        causal: opts[:causal]
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

  # The DeepSeek Sparse Attention (DSA) indexer. It scores every key
  # against every query using its own lightweight projections and returns
  # an additive attention bias, which masks out all but the highest
  # scoring `:index_top_k` keys.
  defp indexer(
         hidden_state,
         query_latent,
         position_ids,
         attention_mask,
         attention_cache,
         offset,
         spec,
         opts
       ) do
    name = opts[:name]

    num_heads = spec.index_num_heads
    head_size = spec.index_head_size

    query =
      query_latent
      |> Axon.dense(num_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query"),
        use_bias: false
      )
      |> Layers.split_heads(num_heads)

    key =
      hidden_state
      |> Axon.dense(head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key"),
        use_bias: false
      )
      |> Axon.layer_norm(epsilon: 1.0e-6, name: join(name, "key_norm"))
      |> Axon.nx(&Nx.new_axis(&1, 2))

    # Note that unlike the main attention, the indexer puts the rotary
    # part of query and key first
    query_rotary = Axon.nx(query, & &1[[.., .., .., 0..(spec.qk_rope_head_size - 1)//1]])
    query_pass = Axon.nx(query, & &1[[.., .., .., spec.qk_rope_head_size..-1//1]])

    key_rotary = Axon.nx(key, & &1[[.., .., .., 0..(spec.qk_rope_head_size - 1)//1]])
    key_pass = Axon.nx(key, & &1[[.., .., .., spec.qk_rope_head_size..-1//1]])

    {query_rotary, key_rotary} =
      Layers.rotary_embedding(
        query_rotary,
        key_rotary,
        position_ids,
        attention_mask,
        spec.qk_rope_head_size,
        name: join(name, "rotary_embedding"),
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base,
        scaling_strategy: spec.rotary_embedding_scaling_strategy,
        interleaved: false
      )

    query = Axon.concatenate([query_rotary, query_pass], axis: -1)

    key =
      [key_rotary, key_pass]
      |> Axon.concatenate(axis: -1)
      |> Axon.nx(&Nx.squeeze(&1, axes: [2]))

    {key, attention_cache} =
      Layers.Decoder.cached_state(key, :indexer_key, attention_cache, offset)

    head_weights =
      Axon.dense(hidden_state, num_heads,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "head_weights"),
        use_bias: false
      )

    bias =
      Axon.layer(
        &indexer_bias_impl/6,
        [query, key, head_weights, Axon.optional(attention_mask), Axon.optional(offset)],
        op_name: :sparse_attention_indexer,
        index_top_k: spec.index_top_k,
        scale: 1.0 / :math.sqrt(head_size)
      )

    {bias, attention_cache}
  end

  defnp indexer_bias_impl(query, key, head_weights, attention_mask, offset, opts \\ []) do
    opts = keyword!(opts, [:index_top_k, :scale, mode: :inference])

    query = Nx.as_type(query, :f32)
    key = Nx.as_type(key, :f32)

    # {batch_size, sequence_length, num_heads, key_sequence_length}
    scores = Nx.dot(query, [3], [0], key, [2], [0])
    scores = Axon.Activations.relu(scores * opts[:scale])

    head_weights = Nx.as_type(head_weights, :f32) / Nx.sqrt(Nx.axis_size(query, 2))

    # {batch_size, sequence_length, key_sequence_length}
    index_scores = Nx.sum(scores * Nx.new_axis(head_weights, -1), axes: [2])

    query_length = Nx.axis_size(query, 1)
    key_length = Nx.axis_size(key, 1)

    offset =
      case offset do
        %Axon.None{} -> 0
        offset -> offset
      end

    mask = Nx.greater_equal(Nx.iota({query_length, 1}) + offset, Nx.iota({1, key_length}))

    mask =
      case attention_mask do
        %Axon.None{} ->
          Nx.new_axis(mask, 0)

        attention_mask ->
          Nx.new_axis(mask, 0) and Nx.new_axis(attention_mask, 1)
      end

    index_scores = Nx.select(mask, index_scores, Nx.Constants.neg_infinity(:f32))

    top_k = min(opts[:index_top_k], key_length)

    order = Nx.argsort(index_scores, axis: -1, direction: :desc, stable: true)
    rank = Nx.argsort(order, axis: -1, stable: true)
    selected = Nx.less(rank, top_k)

    selected
    |> Nx.select(Nx.tensor(0.0, type: :f32), Nx.Constants.min_finite(:f32))
    |> Nx.new_axis(1)
  end

  defp block_ffn(block_idx, spec) do
    if block_idx < spec.first_dense_blocks do
      &Moe.gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
        activation: spec.activation,
        kernel_initializer: kernel_initializer(spec),
        name: &2
      )
    else
      &Moe.block(&1,
        num_experts: spec.num_experts,
        num_experts_per_token: spec.num_experts_per_token,
        hidden_size: spec.hidden_size,
        intermediate_size: spec.moe_intermediate_size,
        shared_expert_intermediate_size:
          spec.num_shared_experts && spec.num_shared_experts * spec.moe_intermediate_size,
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
        kernel_initializer: kernel_initializer(spec),
        name: &2
      )
    end
  end

  defp qk_head_size(spec), do: spec.qk_nope_head_size + spec.qk_rope_head_size

  defp attention_scale(spec), do: 1.0 / :math.sqrt(qk_head_size(spec))

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

      data = data |> Shared.text_config() |> Shared.normalize_rope_options()

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", number()},
          moe_intermediate_size: {"moe_intermediate_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          query_lora_rank: {"q_lora_rank", number()},
          key_value_lora_rank: {"kv_lora_rank", number()},
          qk_nope_head_size: {"qk_nope_head_dim", number()},
          qk_rope_head_size: {"qk_rope_head_dim", number()},
          value_head_size: {"v_head_dim", number()},
          num_experts: {"n_routed_experts", number()},
          num_experts_per_token: {"num_experts_per_tok", number()},
          num_shared_experts: {"n_shared_experts", optional(number())},
          num_expert_groups: {"n_group", number()},
          num_expert_groups_per_token: {"topk_group", number()},
          first_dense_blocks: {"first_k_dense_replace", number()},
          routed_scaling_factor: {"routed_scaling_factor", number()},
          normalize_top_k_probabilities: {"norm_topk_prob", boolean()},
          activation: {"hidden_act", activation()},
          rotary_embedding_base: {"rope_theta", number()},
          rotary_embedding_scaling_strategy: {"rope_scaling", optional(rope_scaling_strategy())},
          rotary_embedding_interleaved: {"rope_interleave", boolean()},
          index_top_k: {"index_topk", number()},
          index_num_heads: {"index_n_heads", number()},
          index_head_size: {"index_head_dim", number()},
          use_attention_bias: {"attention_bias", boolean()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.Layers.Moe

    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.self_attention.query_down" => "model.layers.{n}.self_attn.q_a_proj",
        "decoder.blocks.{n}.self_attention.indexer.query" =>
          "model.layers.{n}.self_attn.indexer.wq_b",
        "decoder.blocks.{n}.self_attention.indexer.key" =>
          "model.layers.{n}.self_attn.indexer.wk",
        "decoder.blocks.{n}.self_attention.indexer.key_norm" =>
          "model.layers.{n}.self_attn.indexer.k_norm",
        "decoder.blocks.{n}.self_attention.indexer.head_weights" =>
          "model.layers.{n}.self_attn.indexer.weights_proj",
        "decoder.blocks.{n}.self_attention.query_norm" =>
          "model.layers.{n}.self_attn.q_a_layernorm",
        "decoder.blocks.{n}.self_attention.query_up" => "model.layers.{n}.self_attn.q_b_proj",
        "decoder.blocks.{n}.self_attention.key_value_down" =>
          "model.layers.{n}.self_attn.kv_a_proj_with_mqa",
        "decoder.blocks.{n}.self_attention.key_value_norm" =>
          "model.layers.{n}.self_attn.kv_a_layernorm",
        "decoder.blocks.{n}.self_attention.key_value_up" =>
          "model.layers.{n}.self_attn.kv_b_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.post_attention_layernorm",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head"),
        "sequence_classification_head.output" => "score"
      }
      |> Map.merge(
        Moe.params_mapping("decoder.blocks.{n}.ffn", "model.layers.{n}.mlp",
          num_experts: spec.num_experts,
          intermediate_size: spec.moe_intermediate_size,
          shared_expert: spec.num_shared_experts != nil
        )
      )
      |> Bumblebee.HuggingFace.Transformers.Utils.expand_params_mapping_source_layer_names(fn
        "model.layers." <> rest = name -> [name, "model.blocks." <> rest]
        name -> [name]
      end)
    end
  end
end
