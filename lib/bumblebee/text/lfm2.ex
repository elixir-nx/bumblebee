defmodule Bumblebee.Text.Lfm2 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 128_000,
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
        default: 2048,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 10_752,
        doc: "the dimensionality of intermediate layers"
      ],
      num_blocks: [
        default: 30,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 8,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      block_types: [
        default: nil,
        doc: """
        a list with the type of each block, either `:full_attention` for blocks using attention,
        or `:conv` for blocks using a gated short convolution. When `nil`, every block uses
        attention
        """
      ],
      conv_kernel_size: [
        default: 3,
        doc: "the kernel size of the causal short convolutions"
      ],
      use_conv_bias: [
        default: false,
        doc: "whether to use bias in the short convolution blocks"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 1_000_000,
        doc: "base for computing rotary embedding frequency"
      ],
      rotary_embedding_scaling_strategy: [
        default: nil,
        doc: """
        scaling configuration for rotary embedding. Currently the supported values are:

          * `%{type: :linear, factor: number()}`

          * `%{type: :dynamic, factor: number()}`

          * `%{type: :yarn, factor: number(), original_max_positions: number()}`

        """
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
        default: true,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: nil)

  @moduledoc """
  LFM2 model family.

  LFM2 is a hybrid model, where most blocks replace attention with a
  gated causal short convolution and the remaining blocks use regular
  grouped-query attention with normalized query and key.

  ## Architectures

    * `:base` - plain LFM2 without any head on top

    * `:for_causal_language_modeling` - LFM2 with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - LFM2 with a sequence classification
      head. The head returns logits corresponding to possible classes

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

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [:base, :for_causal_language_modeling, :for_sequence_classification]

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
    attention_head_size = div(spec.hidden_size, spec.num_attention_heads)
    window_size = spec.conv_kernel_size - 1

    attention_block? = fn idx -> Enum.at(block_types, idx) == :full_attention end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      # The convolution blocks have no key-value cache, so we keep a
      # placeholder of minimal size for them
      decoder_num_attention_heads: fn idx ->
        if attention_block?.(idx), do: spec.num_attention_heads, else: 1
      end,
      attention_head_size: fn idx ->
        if attention_block?.(idx), do: attention_head_size, else: 1
      end,
      decoder_num_blocks: spec.num_blocks,
      extra_window_states: fn idx ->
        if attention_block?.(idx) do
          []
        else
          [convolution: {window_size, spec.hidden_size}]
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
              |> Nx.not_equal(spec.pad_token_id || 0)
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
        Axon.embedding(inputs["input_ids"], spec.vocab_size, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: "embedder.token_embedding"
        )
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
        epsilon: spec.layer_norm_epsilon
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  # The blocks alternate between attention and short convolution, and the
  # convolutions have their own cache entries, so we build the blocks
  # explicitly, rather than using `Bumblebee.Layers.Transformer.blocks/2`
  defp decoder(hidden_state, position_ids, attention_mask, cache, spec, opts) do
    name = opts[:name]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    block_types = block_types(spec)

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
              name: join(block_name, "operator_norm"),
              epsilon: spec.layer_norm_epsilon
            )

          {hidden_state, attention, attention_cache} =
            case Enum.at(block_types, idx) do
              :full_attention ->
                attention(
                  hidden_state,
                  position_ids,
                  attention_mask,
                  attention_cache,
                  offset,
                  spec,
                  name: join(block_name, "self_attention")
                )

              :conv ->
                {hidden_state, attention_cache} =
                  short_convolution(
                    hidden_state,
                    attention_mask,
                    attention_cache,
                    offset,
                    spec,
                    name: join(block_name, "convolution")
                  )

                {hidden_state, Layers.none(), attention_cache}
            end

          hidden_state = Axon.add(hidden_state, shortcut)

          hidden_state =
            hidden_state
            |> Layers.rms_norm(
              name: join(block_name, "output_norm"),
              epsilon: spec.layer_norm_epsilon
            )
            |> gated_ffn(spec.intermediate_size, spec.hidden_size,
              activation: spec.activation,
              name: join(block_name, "ffn")
            )
            |> Axon.add(hidden_state)

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

  defp attention(hidden_state, position_ids, attention_mask, attention_cache, offset, spec, opts) do
    name = opts[:name]

    num_heads = spec.num_attention_heads
    num_key_value_heads = spec.num_key_value_heads
    head_size = div(spec.hidden_size, num_heads)

    query =
      hidden_state
      |> Axon.dense(num_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query"),
        use_bias: false
      )
      |> Layers.split_heads(num_heads)
      |> Layers.rms_norm(name: join(name, "query_norm"), epsilon: spec.layer_norm_epsilon)

    key =
      hidden_state
      |> Axon.dense(num_key_value_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key"),
        use_bias: false
      )
      |> Layers.split_heads(num_key_value_heads)
      |> Layers.rms_norm(name: join(name, "key_norm"), epsilon: spec.layer_norm_epsilon)

    value =
      hidden_state
      |> Axon.dense(num_key_value_heads * head_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "value"),
        use_bias: false
      )
      |> Layers.split_heads(num_key_value_heads)

    {query, key} =
      Layers.rotary_embedding(query, key, position_ids, attention_mask, head_size,
        name: join(name, "rotary_embedding"),
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base,
        scaling_strategy: spec.rotary_embedding_scaling_strategy
      )

    num_key_value_groups = div(num_heads, num_key_value_heads)
    key = repeat_states(key, num_key_value_groups)
    value = repeat_states(value, num_key_value_groups)

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, attention_cache, offset)

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        attention_mask,
        Layers.none(),
        Layers.none(),
        offset,
        causal: true
      )

    attention_output =
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output"),
        use_bias: false
      )

    {attention_output, attention_weights, attention_cache}
  end

  defp repeat_states(state, 1), do: state
  defp repeat_states(state, times), do: Layers.repeat_interleave(state, times, axis: 2)

  # A gated causal short convolution. The input is projected into three
  # parts, two of which gate the convolution input and output
  defp short_convolution(hidden_state, attention_mask, attention_cache, offset, spec, opts) do
    name = opts[:name]

    projected =
      Axon.dense(hidden_state, 3 * spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "input"),
        use_bias: spec.use_conv_bias
      )

    input_gate = Axon.nx(projected, & &1[[.., .., 0..(spec.hidden_size - 1)//1]])

    output_gate =
      Axon.nx(projected, & &1[[.., .., spec.hidden_size..(2 * spec.hidden_size - 1)//1]])

    input = Axon.nx(projected, & &1[[.., .., (2 * spec.hidden_size)..-1//1]])

    hidden_state = Axon.multiply(input_gate, input)

    {full_hidden_state, attention_cache} =
      Layers.Decoder.cached_window_state(hidden_state, :convolution, attention_cache)

    output =
      hidden_state
      |> Layers.causal_depthwise_conv1d(full_hidden_state, attention_mask, offset,
        channels: spec.hidden_size,
        kernel_size: spec.conv_kernel_size,
        use_bias: spec.use_conv_bias,
        name: join(name, "convolution")
      )
      |> then(&Axon.multiply(output_gate, &1))
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output"),
        use_bias: spec.use_conv_bias
      )

    {output, attention_cache}
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]

    gate = Axon.dense(hidden_state, intermediate_size, name: join(name, "gate"), use_bias: false)

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    hidden_state = Axon.multiply(Layers.activation(gate, opts[:activation]), intermediate)

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
  end

  defp block_types(spec) do
    spec.block_types || List.duplicate(:full_attention, spec.num_blocks)
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

      block_type_converter =
        list(mapping(%{"full_attention" => :full_attention, "conv" => :conv}))

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
          block_types: {"layer_types", optional(block_type_converter)},
          conv_kernel_size: {"conv_L_cache", number()},
          use_conv_bias: {"conv_bias", boolean()},
          rotary_embedding_base: {"rope_theta", number()},
          rotary_embedding_scaling_strategy: {"rope_scaling", optional(rope_scaling_strategy())},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      # Some checkpoints specify the attention blocks as a list of
      # indices, rather than a list of block types
      opts =
        case {opts[:block_types], data["full_attn_idxs"]} do
          {nil, indices} when is_list(indices) ->
            num_blocks = opts[:num_blocks] || spec.num_blocks

            block_types =
              for idx <- 0..(num_blocks - 1) do
                if idx in indices, do: :full_attention, else: :conv
              end

            Keyword.put(opts, :block_types, block_types)

          _other ->
            opts
        end

      # Older checkpoints derive the feed-forward size from the hidden
      # size, rather than specifying it directly
      opts =
        if data["block_auto_adjust_ff_dim"] do
          Keyword.put(opts, :intermediate_size, adjusted_intermediate_size(data, opts))
        else
          opts
        end

      @for.config(spec, opts)
    end

    defp adjusted_intermediate_size(data, opts) do
      intermediate_size = div(2 * opts[:intermediate_size], 3)

      case data["block_ffn_dim_multiplier"] do
        nil ->
          intermediate_size

        multiplier ->
          multiple_of = data["block_multiple_of"] || 256
          intermediate_size = trunc(multiplier * intermediate_size)
          multiple_of * div(intermediate_size + multiple_of - 1, multiple_of)
      end
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.out_proj",
        "decoder.blocks.{n}.self_attention.query_norm" =>
          "model.layers.{n}.self_attn.q_layernorm",
        "decoder.blocks.{n}.self_attention.key_norm" => "model.layers.{n}.self_attn.k_layernorm",
        "decoder.blocks.{n}.convolution.input" => "model.layers.{n}.conv.in_proj",
        "decoder.blocks.{n}.convolution.output" => "model.layers.{n}.conv.out_proj",
        "decoder.blocks.{n}.convolution.convolution" => %{
          "kernel" => {
            [{"model.layers.{n}.conv.conv", "weight"}],
            fn [kernel] -> Nx.squeeze(kernel, axes: [1]) end
          },
          "bias" => {
            [{"model.layers.{n}.conv.conv", "bias"}],
            fn [bias] -> bias end
          }
        },
        "decoder.blocks.{n}.operator_norm" => "model.layers.{n}.operator_norm",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.ffn_norm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.feed_forward.w1",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.feed_forward.w3",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.feed_forward.w2",
        "output_norm" => "model.embedding_norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head"),
        "sequence_classification_head.output" => "score"
      }
    end
  end
end
