defmodule Bumblebee.Text.Gpt2 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50257,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 1024,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Transformer blocks in the decoder"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      intermediate_size: [
        default: nil,
        doc: """
        the dimensionality of the intermediate (often named feed-forward) layer in the decoder.
        If not specified, defaults to 4 times `:hidden_size`
        """
      ],
      activation: [
        default: :gelu_new,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embedding and encoder"
      ],
      embeddings_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embeddings"
      ],
      attention_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for attention weights"
      ],
      classifier_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for the classification head"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([
        :use_cross_attention,
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ]) ++
      Shared.token_options(
        bos_token_id: 50256,
        eos_token_id: 50256,
        pad_token_id: 50256
      ) ++ Shared.generation_options()

  @moduledoc """
  GPT-2 model family.

  ## Architectures

    * `:base` - plain GPT-2 without any head on top

    * `:for_causal_language_modeling` - GPT-2 with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - GPT-2 with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - GPT-2 with a token classification
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, seq_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, seq_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"token_type_ids"` - `{batch_size, seq_length}`

      Mask distinguishing groups in the input sequence. This is used
      in when the input sequence is a semantically a pair of sequences.

    * `"position_ids"` - `{batch_size, seq_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeds"` - `{batch_size, seq_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeds"` are present,
      then `"input_ids"` will be ignored.

    * `"encoder_last_hidden_state"` - `{batch_size, encoder_seq_length, hidden_size}`

      Last hidden state output from the encoder. This hidden state is
      used in cross-attention blocks in the decoder. If specified, the
      model will skip the encoding process and use this value directly
      for cross-attentions in the decoder.

    * `"encoder_attention_mask"` - `{batch_size, encoder_seq_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"cross_attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the cross-attention blocks in
      the decoder with shape.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  @impl true
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling,
      :for_sequence_classification,
      :for_token_classification
    ]

  @impl true
  def config(spec, opts \\ []) do
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
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> gpt2(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)

    outputs = gpt2(inputs, spec, name: "transformer")

    # TODO: Tie lm-head to word embedding as a spec option
    logits =
      Layers.dense_transposed(outputs.last_hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: "transformer.wte"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = spec) do
    inputs = inputs(spec)

    outputs = gpt2(inputs, spec, name: "transformer")

    logits =
      outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(spec))
      |> Axon.dense(spec.num_labels, name: "classifier")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)

    outputs = gpt2(inputs, spec, name: "transformer")

    logits =
      outputs.last_hidden_state
      |> Layers.dense_transposed(spec.num_labels, name: "score")

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
      cross_attentions: outputs.cross_attentions
    })
  end

  @impl true
  def init_cache(spec, batch_size, max_length, inputs) do
    encoder_sequence_length =
      if encoder_last_hidden_state = inputs["encoder_last_hidden_state"] do
        Nx.axis_size(encoder_last_hidden_state, 1)
      end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      num_decoder_attention_heads: spec.num_attention_heads,
      num_encoder_attention_heads: spec.num_attention_heads,
      num_decoder_blocks: spec.num_blocks,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  defp gpt2(inputs, spec, opts \\ []) do
    name = opts[:name]

    input_embeds =
      Layers.default inputs["input_embeds"] do
        Axon.embedding(inputs["input_ids"], spec.vocab_size, spec.hidden_size,
          name: join(name, "wte")
        )
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(input_embeds)
      end

    position_embeds =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size, name: join(name, "wpe"))

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(input_embeds)
      end

    encoder_attention_mask =
      Layers.default inputs["encoder_attention_mask"] do
        Layers.default_attention_mask(inputs["encoder_last_hidden_state"])
      end

    hidden_state =
      input_embeds
      |> Axon.add(position_embeds)
      |> Axon.dropout(rate: spec.embeddings_dropout_rate)

    block_outputs =
      blocks(
        hidden_state,
        attention_mask,
        inputs["head_mask"],
        inputs["encoder_last_hidden_state"],
        encoder_attention_mask,
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        name: join(name, "h")
      )

    last_hidden_state =
      Axon.layer_norm(block_outputs.last_hidden_state,
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "ln_f")
      )

    %{
      last_hidden_state: last_hidden_state,
      hidden_states: block_outputs.hidden_states,
      attentions: block_outputs.attentions,
      cross_attentions: block_outputs.cross_attentions,
      cache: block_outputs.cache
    }
  end

  defp blocks(
         hidden_state,
         attention_mask,
         head_mask,
         encoder_last_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      attentions: Layers.maybe_container({}, spec.output_attentions),
      cross_attentions: Layers.maybe_container({}, spec.output_attentions),
      cache: cache
    }

    offset = Layers.Decoder.get_cache_offset(state.cache)

    outputs =
      for idx <- 0..(spec.num_blocks - 1), reduce: state do
        state ->
          block_head_mask = Axon.nx(head_mask, & &1[idx])
          cross_attention_block_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])

          block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

          {hidden_state, attention, cross_attention, block_cache} =
            block(
              state.last_hidden_state,
              attention_mask,
              encoder_last_hidden_state,
              encoder_attention_mask,
              block_head_mask,
              cross_attention_block_head_mask,
              block_cache,
              offset,
              spec,
              name: join(name, idx)
            )

          cache = Layers.Decoder.put_block_cache(state.cache, idx, block_cache)

          %{
            last_hidden_state: hidden_state,
            hidden_states: Layers.append(state.hidden_states, hidden_state),
            attentions: Layers.append(state.attentions, attention),
            cross_attentions: Layers.append(state.cross_attentions, cross_attention),
            cache: cache
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  defp block(
         hidden_state,
         attention_mask,
         encoder_last_hidden_state,
         encoder_attention_mask,
         head_mask,
         cross_attention_head_mask,
         block_cache,
         offset,
         spec,
         opts
       ) do
    name = opts[:name]
    inner_dim = spec.intermediate_size || 4 * spec.hidden_size

    residual = hidden_state

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    {attention_output, attention_weights, self_attention_cache} =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "ln_1")
      )
      |> attention(
        attention_mask,
        nil,
        head_mask,
        self_attention_cache,
        offset,
        spec,
        num_heads: spec.num_attention_heads,
        causal?: true,
        name: join(name, "attn")
      )

    hidden_state = Axon.add(attention_output, residual)

    {hidden_state, cross_attention_weights, cross_attention_cache} =
      if spec.use_cross_attention do
        Layers.if_present encoder_last_hidden_state do
          residual = hidden_state

          {cross_attention_output, cross_attention_weights, cross_attention_cache} =
            hidden_state
            |> Axon.layer_norm(
              channel_index: 2,
              epsilon: spec.layer_norm_epsilon,
              name: join(name, "ln_cross_attn")
            )
            |> attention(
              encoder_attention_mask,
              encoder_last_hidden_state,
              cross_attention_head_mask,
              cross_attention_cache,
              offset,
              spec,
              name: join(name, "crossattention"),
              num_heads: spec.num_attention_heads
            )

          hidden_state = Axon.add(cross_attention_output, residual)
          {hidden_state, cross_attention_weights, cross_attention_cache}
        else
          {hidden_state, Layers.none(), cross_attention_cache}
        end
      else
        {hidden_state, Layers.none(), cross_attention_cache}
      end

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "ln_2")
      )
      |> mlp(inner_dim, spec, name: join(name, "mlp"))
      |> Axon.add(residual)

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, attention_weights, cross_attention_weights, block_cache}
  end

  defp attention(
         hidden_state,
         attention_mask,
         cross_hidden_state,
         block_head_mask,
         attention_cache,
         offset,
         spec,
         opts
       ) do
    name = opts[:name]
    num_heads = opts[:num_heads]
    causal? = Keyword.get(opts, :causal?, false)
    cross_attention? = cross_hidden_state != nil

    {query, key, value} =
      if cross_attention? do
        q_out =
          Layers.conv1d(hidden_state, spec.hidden_size,
            kernel_initializer: kernel_initializer(spec),
            name: join(name, "q_attn")
          )

        {query} = Axon.split(q_out, 1, axis: 1)

        kv_out =
          Layers.conv1d(cross_hidden_state, spec.hidden_size * 2,
            kernel_initializer: kernel_initializer(spec),
            name: join(name, "c_attn")
          )

        {key, value} = Axon.split(kv_out, 2, axis: 2)

        {query, key, value}
      else
        qkv_out =
          Layers.conv1d(hidden_state, spec.hidden_size * 3,
            kernel_initializer: kernel_initializer(spec),
            name: join(name, "c_attn")
          )

        Axon.split(qkv_out, 3, axis: 2)
      end

    query = Layers.split_heads(query, num_heads)
    key = Layers.split_heads(key, num_heads)
    value = Layers.split_heads(value, num_heads)

    attention_mask = Layers.expand_attention_mask(attention_mask)

    attention_mask =
      if causal? do
        Layers.Decoder.apply_causal_mask(attention_mask, query, offset)
      else
        attention_mask
      end

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, attention_cache, offset,
        cross_attention?: cross_attention?
      )

    attention_bias = Layers.attention_bias(attention_mask)

    attention_weights = Layers.attention_weights(query, key, attention_bias)

    attention_weights =
      attention_weights
      |> Axon.dropout(rate: spec.attention_dropout_rate)
      |> Layers.apply_attention_head_mask(block_head_mask)

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()
      |> Layers.conv1d(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "c_proj")
      )
      |> Axon.dropout(rate: spec.dropout_rate)

    {attention_output, attention_weights, attention_cache}
  end

  defp mlp(hidden_state, inner_dim, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.conv1d(inner_dim,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "c_fc")
    )
    |> Layers.activation(spec.activation, name: join(name, "act"))
    |> Layers.conv1d(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "c_proj")
    )
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}
    decoder_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("head_mask", optional: true, shape: decoder_head_mask_shape),
      Axon.input("input_embeds", optional: true, shape: hidden_shape),
      Axon.input("encoder_last_hidden_state", optional: true, shape: hidden_shape),
      Axon.input("encoder_attention_mask", optional: true, shape: shape),
      Axon.input("cross_attention_head_mask", optional: true, shape: decoder_head_mask_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp classifier_dropout_rate(spec) do
    spec.classifier_dropout_rate || spec.hidden_dropout
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
          max_positions: {"n_position", number()},
          hidden_size: {"n_embd", number()},
          num_blocks: {"n_layer", number()},
          num_attention_heads: {"n_head", number()},
          intermediate_size: {"n_inner", optional(number())},
          activation: {"activation_function", atom()},
          dropout_rate: {"resid_pdrop", number()},
          embeddings_dropout_rate: {"embd_pdrop", number()},
          attention_dropout_rate: {"attn_pdrop", number()},
          classifier_dropout_rate: {"classifier_dropout", number()},
          layer_norm_epsilon: {"layer_norm_epsilon", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
