defmodule Bumblebee.Text.Roberta do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 30522,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 512,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      max_token_types: [
        default: 2,
        doc: """
        the vocabulary size of the token type embedding (also referred to as segment embedding).
        This corresponds to how many different token groups can be distinguished in the input
        """
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 3072,
        doc:
          "the dimensionality of the intermediate (often named feed-forward) layer in the encoder"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embedding and encoder"
      ],
      attention_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for attention weights"
      ],
      classifier_dropout_rate: [
        default: nil,
        doc:
          "the dropout rate for the classification head. If not specified, the value of `:dropout_rate` is used instead"
      ],
      layer_norm_epsilon: [
        default: 1.0e-12,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_range: [
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
      Shared.token_options(pad_token_id: 1, bos_token_id: 0, eos_token_id: 2) ++
      Shared.generation_options()

  @moduledoc """
  RoBERTa model family.

  ## Architectures

    * `:base` - plain RoBERTa without any head on top

    * `:for_masked_language_modeling` - RoBERTa with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - RoBERTa with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - RoBERTa with a token classification
      head. The head returns logits for each token in the original
      sequence

    * `:for_question_answering` - RoBERTa with a span classification head.
      The head returns logits for the span start and end positions

    * `:for_multiple_choice` - RoBERTa with a multiple choice prediction
      head. Each input in the batch consists of several sequences to
      choose from and the model returns logits corresponding to those
      choices

    * `:for_causal_language_modeling` - RoBERTa working as a decoder with
      a language modeling head. The head returns logits for each token in
      the original sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"token_type_ids"` - `{batch_size, sequence_length}`

      Mask distinguishing groups in the input sequence. This is used
      in when the input sequence is a semantically a pair of sequences.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

  ### Exceptions

  The `:for_multiple_choice` model accepts groups of sequences, so the
  expected sequence shape is `{batch_size, num_choices, sequence_length}`.

  The `:for_causal_language_modeling` model is a decoder and accepts
  the following additional inputs: `"encoder_hidden_state"`,
  `"encoder_attention_mask"`, `"cross_attention_head_mask"`, `"cache"`.

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
      :for_masked_language_modeling,
      :for_sequence_classification,
      :for_token_classification,
      :for_question_answering,
      :for_multiple_choice,
      :for_causal_language_modeling
    ]

  @impl true
  def config(spec, opts \\ []) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(%{architecture: :for_multiple_choice}) do
    %{"input_ids" => Nx.template({1, 1, 1}, :s64)}
  end

  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs(spec)
    |> roberta(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_masked_language_modeling} = spec) do
    outputs = inputs(spec) |> roberta(spec, name: "roberta")

    logits = lm_prediction_head(outputs.hidden_state, spec, name: "lm_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    outputs = inputs(spec) |> roberta(spec, name: "roberta")

    logits =
      outputs.hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(spec), name: "dropout")
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "classifier.dense"
      )
      |> Axon.tanh()
      |> Axon.dropout(rate: classifier_dropout_rate(spec), name: "dropout")
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "classifier.out_proj"
      )

    Layers.output(%{
      logits: Axon.nx(logits, fn x -> x[[0..-1//1, 0, 0..-1//1]] end),
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = spec) do
    outputs = inputs(spec) |> roberta(spec, name: "roberta")

    logits =
      outputs.hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(spec), name: "dropout")
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "classifier"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_question_answering} = spec) do
    outputs = inputs(spec) |> roberta(spec, name: "roberta")

    logits =
      outputs.hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(spec), name: "dropout")
      |> Axon.dense(2,
        kernel_initializer: kernel_initializer(spec),
        name: "qa_outputs"
      )

    {start_logits, end_logits} = Layers.split_pair(logits)

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_multiple_choice} = spec) do
    inputs = inputs(spec, shape: {nil, nil, nil})

    group_inputs = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]

    flat_inputs =
      Enum.reduce(group_inputs, inputs, fn name, inputs ->
        Map.update!(inputs, name, &Layers.flatten_leading/1)
      end)

    outputs = roberta(flat_inputs, spec, name: "roberta")

    logits =
      outputs.pooled_state
      |> Axon.dropout(rate: classifier_dropout_rate(spec), name: "dropout")
      |> Axon.dense(1,
        kernel_initializer: kernel_initializer(spec),
        name: "classifier"
      )

    # The final shape depends on the dynamic batch size and number
    # of choices, so we do a custom reshape at runtime
    logits =
      Axon.layer(
        fn logits, input_ids, _opts ->
          num_choices = Nx.axis_size(input_ids, 1)
          Nx.reshape(logits, {:auto, num_choices})
        end,
        [logits, inputs["input_ids"]]
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    outputs = inputs(spec, decoder?: true) |> roberta(spec, decoder?: true, name: "roberta")

    logits = lm_prediction_head(outputs.hidden_state, spec, name: "lm_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
    })
  end

  @impl true
  def init_cache(spec, batch_size, max_length, inputs) do
    encoder_sequence_length =
      if encoder_hidden_state = inputs["encoder_hidden_state"] do
        Nx.axis_size(encoder_hidden_state, 1)
      end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      decoder_attention_heads: spec.num_attention_heads,
      encoder_attention_heads: spec.num_attention_heads,
      decoder_blocks: spec.num_blocks,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  defp inputs(spec, opts \\ []) do
    shape = Keyword.get(opts, :shape, {nil, nil})
    decoder? = Keyword.get(opts, :decoder?, false)

    hidden_shape = Tuple.append(shape, spec.hidden_size)
    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", shape: shape),
        Axon.input("attention_mask", optional: true, shape: shape),
        Axon.input("token_type_ids", optional: true, shape: shape),
        Axon.input("position_ids", optional: true, shape: shape),
        Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape)
      ])

    extra_decoder_inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
        Axon.input("encoder_attention_mask", optional: true, shape: shape),
        Axon.input("cross_attention_head_mask", optional: true, shape: attention_head_mask_shape),
        Axon.input("cache", optional: true)
      ])

    extra_decoder_inputs =
      if decoder? do
        extra_decoder_inputs
      else
        Map.new(extra_decoder_inputs, fn {name, _input} -> {name, Layers.none()} end)
      end

    Map.merge(inputs, extra_decoder_inputs)
  end

  defp roberta(inputs, spec, opts \\ []) do
    name = opts[:name]
    decoder? = Keyword.get(opts, :decoder?, false)

    input_ids = inputs["input_ids"]

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(input_ids)
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        # Position numbers begin at padding token id + 1
        Layers.default_position_ids(input_ids, offset: spec.pad_token_id + 1)
      end

    token_type_ids =
      Layers.default inputs["token_type_ids"] do
        Layers.default_token_type_ids(input_ids)
      end

    encoder_attention_mask =
      Layers.default inputs["encoder_attention_mask"] do
        Layers.default_attention_mask(inputs["encoder_hidden_state"])
      end

    hidden_state =
      embeddings(input_ids, position_ids, token_type_ids, spec, name: join(name, "embeddings"))

    encoder_outputs =
      encoder(
        hidden_state,
        attention_mask,
        inputs["attention_head_mask"],
        inputs["encoder_hidden_state"],
        encoder_attention_mask,
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        decoder?: decoder?,
        name: join(name, "encoder")
      )

    pooled_state = pooler(encoder_outputs.hidden_state, spec, name: join(name, "pooler"))

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      cross_attentions: encoder_outputs.cross_attentions,
      cache: encoder_outputs.cache
    }
  end

  defp embeddings(input_ids, position_ids, token_type_ids, spec, opts) do
    name = opts[:name]

    inputs_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "word_embeddings")
      )

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "position_embeddings")
      )

    token_type_embeddings =
      Axon.embedding(token_type_ids, spec.max_token_types, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_type_embeddings")
      )

    Axon.add([inputs_embeddings, position_embeddings, token_type_embeddings])
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "LayerNorm"))
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp encoder(
         hidden_state,
         attention_mask,
         attention_head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]
    decoder? = opts[:decoder?]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)

    outputs =
      encoder_blocks(
        hidden_state,
        attention_mask,
        attention_head_mask,
        encoder_hidden_state,
        encoder_attention_mask,
        cross_attention_head_mask,
        cache,
        spec,
        decoder?: decoder?,
        name: join(name, "layer")
      )

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  defp encoder_blocks(
         hidden_state,
         attention_mask,
         attention_head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]
    decoder? = opts[:decoder?]

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      attentions: Layers.maybe_container({}, spec.output_attentions),
      cross_attentions: Layers.maybe_container({}, spec.output_attentions),
      cache: cache
    }

    offset = Layers.Decoder.get_cache_offset(state.cache)

    for idx <- 0..(spec.num_blocks - 1), reduce: state do
      state ->
        block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])
        cross_attention_block_attention_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])

        block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

        {hidden_state, attention, cross_attention, block_cache} =
          roberta_block(
            state.hidden_state,
            attention_mask,
            block_attention_head_mask,
            encoder_hidden_state,
            encoder_attention_mask,
            cross_attention_block_attention_head_mask,
            block_cache,
            offset,
            spec,
            decoder?: decoder?,
            name: join(name, idx)
          )

        cache = Layers.Decoder.put_block_cache(state.cache, idx, block_cache)

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention),
          cross_attentions: Layers.append(state.cross_attentions, cross_attention),
          cache: cache
        }
    end
  end

  defp roberta_block(
         hidden_state,
         attention_mask,
         block_attention_head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_block_attention_head_mask,
         block_cache,
         offset,
         spec,
         opts
       ) do
    name = opts[:name]
    decoder? = opts[:decoder?]

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    {attention_output, attention, self_attention_cache} =
      attention(
        hidden_state,
        attention_mask,
        nil,
        block_attention_head_mask,
        self_attention_cache,
        offset,
        spec,
        causal?: decoder?,
        name: join(name, "attention")
      )

    {attention_output, cross_attention, cross_attention_cache} =
      if decoder? and spec.use_cross_attention do
        Layers.if_present encoder_hidden_state do
          attention(
            attention_output,
            encoder_attention_mask,
            encoder_hidden_state,
            cross_attention_block_attention_head_mask,
            cross_attention_cache,
            offset,
            spec,
            name: join(name, "crossattention")
          )
        else
          {attention_output, Layers.none(), cross_attention_cache}
        end
      else
        {attention_output, Layers.none(), cross_attention_cache}
      end

    hidden_state = intermediate(attention_output, spec, name: join(name, "intermediate"))
    hidden_state = output(hidden_state, attention_output, spec, name: join(name, "output"))

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, attention, cross_attention, block_cache}
  end

  defp attention(
         hidden_state,
         attention_mask,
         cross_hidden_state,
         block_attention_head_mask,
         attention_cache,
         offset,
         spec,
         opts
       ) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    {attention_output, attention, block_cache} =
      self_attention(
        hidden_state,
        attention_mask,
        cross_hidden_state,
        block_attention_head_mask,
        attention_cache,
        offset,
        spec,
        causal?: causal?,
        name: join(name, "self")
      )

    hidden_state = self_output(attention_output, hidden_state, spec, name: join(name, "output"))

    {hidden_state, attention, block_cache}
  end

  defp self_attention(
         hidden_state,
         attention_mask,
         cross_hidden_state,
         block_attention_head_mask,
         attention_cache,
         offset,
         spec,
         opts
       ) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)
    cross_attention? = cross_hidden_state != nil

    num_heads = spec.num_attention_heads

    query =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query")
      )
      |> Layers.split_heads(num_heads)

    # For cross-attention we are given encoder hidden state
    projection_states = cross_hidden_state || hidden_state

    value =
      projection_states
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "value")
      )
      |> Layers.split_heads(num_heads)

    key =
      projection_states
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key")
      )
      |> Layers.split_heads(num_heads)

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

    attention_weights =
      Layers.attention_weights(query, key, attention_bias)
      |> Axon.dropout(rate: spec.attention_dropout_rate, name: join(name, "dropout"))
      |> Layers.apply_attention_head_mask(block_attention_head_mask)

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()

    {attention_output, attention_weights, attention_cache}
  end

  defp self_output(hidden_state, input, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
    |> Axon.add(input)
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "LayerNorm"))
  end

  defp intermediate(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.intermediate_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Layers.activation(spec.activation, name: join(name, "activation"))
  end

  defp output(hidden_state, attention_output, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
    |> Axon.add(attention_output)
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "LayerNorm"))
  end

  defp pooler(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(index: 0, axis: 1, name: join(name, "head"))
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Axon.tanh()
  end

  defp lm_prediction_head(hidden_state, spec, opts) do
    name = opts[:name]

    # TODO: use a shared parameter with embeddings.word_embeddings.kernel
    # if spec.tie_word_embeddings is true (relevant for training)

    hidden_state
    |> lm_prediction_head_transform(spec, name: name)
    # We reuse the kernel of input embeddings and add bias for each token
    |> Layers.dense_transposed(spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "decoder")
    )
    |> Axon.bias(name: name)
  end

  defp lm_prediction_head_transform(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Layers.activation(spec.activation, name: join(name, "activation"))
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "layer_norm"))
  end

  defp classifier_dropout_rate(spec) do
    spec.classifier_dropout_rate || spec.dropout_rate
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_range)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          max_token_types: {"type_vocab_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", atom()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          classifier_dropout_rate: {"classifier_dropout", optional(number())},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_range: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
