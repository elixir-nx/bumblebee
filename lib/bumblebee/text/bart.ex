defmodule Bumblebee.Text.Bart do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :num_labels]

  @moduledoc """
  Models based on BART architecture.

  ## Architectures

    * `:base` - plain BART without any head on top

    * `:for_causal_language_modeling` - BART with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_conditional_generation` - BART with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - BART with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_question_answering` - BART with a span classification head.
      The head returns logits for the span start and end positions

  ## Inputs

    * `"input_ids"` - `{batch_size, seq_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, seq_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, seq_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"head_mask"` - `{encoder_layers, encoder_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeds"` - `{batch_size, seq_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeds"` are present,
      then `"input_ids"` will be ignored.

    * `"decoder_input_ids"` - `{batch_size, target_seq_length}`

      Indices of decoder input sequence tokens in the vocabulary. If not
      present and `"input_ids"` is, it will be generated by shifting
      each token in `"input_ids"` to the right once.

    * `"decoder_attention_mask"` - `{batch_size, target_seq_length}`

      Mask indicating which decoder tokens to attend to. This is used
      to ignore padding tokens, which are added when processing a batch
      of sequences with different length.

    * `"decoder_position_ids"` - `{batch_size, target_seq_length}`

      Indices of positions of each decoder input sequence tokens in
      the position embeddings.

    * `"decoder_head_mask"` - `{decoder_layers, decoder_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the decoder.

    * `"decoder_input_embeds"` - `{batch_size, seq_length, hidden_size}`

      Embedded representation of `"decoder_input_ids"`, which can be
      specified for more control over how `"decoder_input_ids"` are
      embedded than the model's internal embedding lookup. If
      `"decoder_input_embeds"` are present, then `"decoder_input_ids"`
      will be ignored.

    * `"encoder_last_hidden_state"` - `{batch_size, seq_length, hidden_size}`

      Last hidden state output from the encoder. This hidden state is
      used in cross-attention blocks in the decoder. If specified, the
      model will skip the encoding process and use this value directly
      for cross-attentions in the decoder.

    * `"cross_attention_head_mask"` - `{decoder_layers, decoder_attention_heads}`

      Mask to nullify selected heads of the cross-attention blocks in
      the decoder with shape.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ### Exceptions

  The `:for_causal_language_modeling` model is just the decoder part and
  accepts the following inputs instead: `"input_ids"`, `"attention_mask"`,
  `"position_ids"`, `"head_mask"`, `"input_embeds"`, `"encoder_last_hidden_state"`,
  `"encoder_attention_mask"`, `"cross_attention_head_mask"`, `"cache"`.

  ## Configuration

    * `:vocab_size` - vocabulary size of the model. Defines the number
      of distinct tokens that can be represented by the in model input
      and output. Defaults to `50265`

    * `:d_model` - dimensionality of the layers and the pooler layer.
      Defaults to `1024`

    * `:encoder_layers` - the number of encoder layers. Defaults to `12`

    * `:decoder_layers` - the number of decoder layers. Defaults to `12`

    * `:encoder_attention_heads` - the number of attention heads in the
      encoder. Defaults to `16`

    * `:decoder_attention_heads` - the number of attention heads in the
      decoder. Defaults to `16`

    * `:encoder_ffn_dim` - dimensionality of the "intermediate" layer in
      the encoder. Defaults to `1024`

    * `:decoder_ffn_dim` - dimensionality of the "intermediate" layer in
      the decoder.

    * `:activation_function` - non-linear activation function in the encoder
      and pooler. Defaults to `:gelu`

    * `:dropout` - dropout probability of all fully-connected layers in
      the embeddings, encoder, and pooler. Defaults to `0.1`

    * `:attention_dropout` - dropout ratio for attention weights.
      Defaults to `0.0`

    * `:activation_dropout` - dropout ratio for activations inside the fully
      connected layer. Defaults to `0.0`

    * `:classifier_dropout` - dropout ratio for classifier. Defaults to `0.0`

    * `:max_position_embeddings` - the maximum sequence length that this
      model might ever be used with. Typically set this to something
      large just in case (e.g. 512 or 1024 or 2048). Defaults to `512`

    * `:init_std` - the standard deviation of the normal
      initializer used for initializing kernel parameters. Defaults
      to `0.02`

    * `:scale_embedding` - scale embeddings by dividing by sqrt(d_model).
      Defaults to `false`

  ### Common options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              vocab_size: 50265,
              max_position_embeddings: 1024,
              encoder_layers: 12,
              encoder_ffn_dim: 4096,
              encoder_attention_heads: 16,
              encoder_layerdrop: 0.0,
              decoder_layers: 12,
              decoder_ffn_dim: 4096,
              decoder_attention_heads: 16,
              decoder_layerdrop: 0.0,
              activation_function: :gelu,
              d_model: 1024,
              dropout: 0.1,
              attention_dropout: 0.0,
              activation_dropout: 0.0,
              init_std: 0.02,
              classifier_dropout: 0.0,
              scale_embedding: false,
              # Tokens
              pad_token_id: 1,
              bos_token_id: 0,
              eos_token_id: 2,
              decoder_start_token_id: 2
            ] ++
              Shared.generation_defaults(
                forced_bos_token_id: 0,
                forced_eos_token_id: 2
              ) ++
              Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Text.Generation

  @impl true
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling,
      :for_conditional_generation,
      :for_sequence_classification,
      :for_question_answering
    ]

  @impl true
  def base_model_prefix(), do: "model"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def input_template(_config) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :for_conditional_generation} = config) do
    inputs = encoder_decoder_inputs(config)
    outputs = bart(inputs, config, name: "model")

    # TODO: Tie lm-head to word embedding as a config option
    lm_logits =
      outputs.last_hidden_state
      |> Layers.dense_transposed(config.vocab_size,
        kernel_initializer: kernel_initializer(config),
        name: "model.shared"
      )

    Layers.output(%{
      logits: lm_logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_last_hidden_state: outputs.encoder_last_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = config) do
    inputs = encoder_decoder_inputs(config)
    outputs = bart(inputs, config, name: "model")

    sentence_representation =
      Axon.layer(
        fn input_ids, hidden_state, _opts ->
          eos_mask = Nx.equal(input_ids, config.eos_token_id)
          eos_idx = Nx.argmax(eos_mask, tie_break: :high, axis: 1)
          Bumblebee.Utils.Nx.batched_take(hidden_state, eos_idx)
        end,
        [inputs["input_ids"], outputs.last_hidden_state]
      )

    logits = classification_head(sentence_representation, config, name: "classification_head")

    Layers.output(%{
      logits: logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_last_hidden_state: outputs.encoder_last_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions
    })
  end

  def model(%__MODULE__{architecture: :for_question_answering} = config) do
    inputs = encoder_decoder_inputs(config)
    outputs = bart(inputs, config, name: "model")

    logits =
      Axon.dense(outputs.last_hidden_state, 2,
        kernel_initializer: kernel_initializer(config),
        name: "qa_outputs"
      )

    {start_logits, end_logits} = Layers.split_pair(logits)

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_last_hidden_state: outputs.encoder_last_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions
    })
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = config) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, config.d_model}
    decoder_head_mask_shape = {config.decoder_layers, config.decoder_attention_heads}

    inputs =
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

    input_embeds =
      Layers.default inputs["input_embeds"] do
        token_embedding(inputs["input_ids"], config, name: "model.decoder.embed_tokens")
      end

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(input_embeds)
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(input_embeds)
      end

    encoder_attention_mask =
      Layers.default inputs["encoder_attention_mask"] do
        Layers.default_attention_mask(inputs["encoder_last_hidden_state"])
      end

    outputs =
      decoder(
        input_embeds,
        attention_mask,
        position_ids,
        inputs["head_mask"],
        inputs["encoder_last_hidden_state"],
        encoder_attention_mask,
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        config,
        name: "model.decoder"
      )

    # TODO: Tie lm-head to word embedding as a config option
    lm_logits =
      outputs.last_hidden_state
      |> Layers.dense_transposed(config.vocab_size,
        kernel_initializer: kernel_initializer(config),
        name: "shared"
      )

    Layers.output(%{
      logits: lm_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :base} = config) do
    inputs = encoder_decoder_inputs(config)

    inputs
    |> bart(config)
    |> Layers.output()
  end

  defp encoder_decoder_inputs(config) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, config.d_model}
    encoder_head_mask_shape = {config.encoder_layers, config.encoder_attention_heads}
    decoder_head_mask_shape = {config.decoder_layers, config.decoder_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("head_mask", optional: true, shape: encoder_head_mask_shape),
      Axon.input("input_embeds", optional: true, shape: hidden_shape),
      Axon.input("decoder_input_ids", optional: true, shape: shape),
      Axon.input("decoder_attention_mask", optional: true, shape: shape),
      Axon.input("decoder_position_ids", optional: true, shape: shape),
      Axon.input("decoder_head_mask", optional: true, shape: decoder_head_mask_shape),
      Axon.input("decoder_input_embeds", optional: true, shape: hidden_shape),
      Axon.input("encoder_last_hidden_state", optional: true, shape: hidden_shape),
      Axon.input("cross_attention_head_mask", optional: true, shape: decoder_head_mask_shape),
      Axon.input("cache", optional: true)
    ])
  end

  @impl true
  def init_cache(config, batch_size, max_length, inputs) do
    encoder_sequence_length =
      if encoder_last_hidden_state = inputs["encoder_last_hidden_state"] do
        Nx.axis_size(encoder_last_hidden_state, 1)
      end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: config.d_model,
      decoder_attention_heads: config.decoder_attention_heads,
      encoder_attention_heads: config.encoder_attention_heads,
      decoder_layers: config.decoder_layers,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  defp bart(inputs, config, opts \\ []) do
    name = opts[:name]

    input_embeds =
      Layers.default inputs["input_embeds"] do
        token_embedding(inputs["input_ids"], config, name: join(name, "shared"))
      end

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(input_embeds)
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(input_embeds)
      end

    decoder_input_embeds =
      Layers.default inputs["decoder_input_embeds"] do
        decoder_input_ids =
          Layers.default inputs["decoder_input_ids"] do
            Layers.shift_tokens_right(inputs["input_ids"], config.decoder_start_token_id)
          end

        token_embedding(decoder_input_ids, config, name: join(name, "shared"))
      end

    decoder_attention_mask =
      Layers.default inputs["decoder_attention_mask"] do
        Layers.default_attention_mask(decoder_input_embeds)
      end

    decoder_position_ids =
      Layers.default inputs["decoder_position_ids"] do
        Layers.default_position_ids(decoder_input_embeds)
      end

    encoder_outputs =
      Layers.if_present inputs["encoder_last_hidden_state"] do
        %{
          last_hidden_state: inputs["encoder_last_hidden_state"],
          hidden_states: Layers.none(),
          attentions: Layers.none()
        }
      else
        encoder(input_embeds, attention_mask, position_ids, inputs["head_mask"], config,
          name: join(name, "encoder")
        )
      end

    decoder_outputs =
      decoder(
        decoder_input_embeds,
        decoder_attention_mask,
        decoder_position_ids,
        inputs["decoder_head_mask"],
        encoder_outputs.last_hidden_state,
        attention_mask,
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        config,
        name: join(name, "decoder")
      )

    %{
      last_hidden_state: decoder_outputs.last_hidden_state,
      decoder_hidden_states: decoder_outputs.hidden_states,
      decoder_attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions,
      cache: decoder_outputs.cache,
      encoder_last_hidden_state: encoder_outputs.last_hidden_state,
      encoder_hidden_states: encoder_outputs.hidden_states,
      encoder_attentions: encoder_outputs.attentions
    }
  end

  defp encoder(input_embeds, attention_mask, position_ids, head_mask, config, opts) do
    name = opts[:name]

    position_embeds = position_embedding(position_ids, config, opts)

    input_embeds
    |> Axon.add(position_embeds)
    |> Axon.layer_norm(channel_index: 2, epsilon: 1.0e-5, name: join(name, "layernorm_embedding"))
    |> Axon.dropout(rate: config.dropout)
    |> encoder_layers(attention_mask, head_mask, config, name: join(name, "layers"))
  end

  defp token_embedding(input_ids, config, opts) do
    name = opts[:name]

    input_embeds =
      Axon.embedding(input_ids, config.vocab_size, config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: name
      )

    if config.scale_embedding do
      Axon.nx(input_embeds, fn x -> Nx.multiply(x, Nx.sqrt(config.d_model)) end)
    else
      input_embeds
    end
  end

  defp position_embedding(position_ids, config, opts) do
    name = opts[:name]

    # For BART we need to offset the embeddings
    offset = 2

    position_ids
    |> Axon.add(Axon.constant(Nx.tensor(offset)))
    |> Axon.embedding(config.max_position_embeddings + offset, config.d_model,
      name: join(name, "embed_positions")
    )
  end

  defp encoder_layers(hidden_state, attention_mask, head_mask, config, opts) do
    name = opts[:name]

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, config.output_hidden_states),
      attentions: Layers.maybe_container({}, config.output_attentions)
    }

    for idx <- 0..(config.encoder_layers - 1), reduce: state do
      state ->
        layer_head_mask = Axon.nx(head_mask, & &1[idx])

        # TODO: wrap encoder layer in a layer_drop combinator

        {hidden_state, attention} =
          encoder_layer(state.last_hidden_state, attention_mask, layer_head_mask, config,
            name: join(name, idx)
          )

        %{
          last_hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention)
        }
    end
  end

  defp encoder_layer(hidden_state, attention_mask, layer_head_mask, config, opts) do
    name = opts[:name]

    residual = hidden_state

    {hidden_state, attention, _} =
      attention(
        hidden_state,
        attention_mask,
        nil,
        layer_head_mask,
        Layers.none(),
        Layers.none(),
        config,
        num_heads: config.encoder_attention_heads,
        name: join(name, "self_attn")
      )

    hidden_state =
      hidden_state
      |> Axon.dropout(rate: config.dropout, name: join(name, "dropout.0"))
      |> Axon.add(residual, name: join(name, "residual.0"))
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: 1.0e-5,
        name: join(name, "self_attn_layer_norm")
      )

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.dense(config.encoder_ffn_dim,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "fc1")
      )
      |> Axon.activation(config.activation_function, name: join(name, "activation"))
      |> Axon.dropout(rate: config.activation_dropout, name: join(name, "dropout.1"))
      |> Axon.dense(config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "fc2")
      )
      |> Axon.add(residual, name: join(name, "residual.1"))
      |> Axon.layer_norm(channel_index: 2, epsilon: 1.0e-5, name: join(name, "final_layer_norm"))

    {hidden_state, attention}
  end

  defp decoder(
         input_embeds,
         attention_mask,
         position_ids,
         head_mask,
         encoder_last_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         config,
         opts
       ) do
    name = opts[:name]

    position_embeds = position_embedding(position_ids, config, opts)

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)

    outputs =
      input_embeds
      |> Axon.add(position_embeds)
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: 1.0e-5,
        name: join(name, "layernorm_embedding")
      )
      |> Axon.dropout(rate: config.dropout)
      |> decoder_layers(
        attention_mask,
        head_mask,
        encoder_last_hidden_state,
        encoder_attention_mask,
        cross_attention_head_mask,
        cache,
        config,
        name: join(name, "layers")
      )

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, input_embeds))
  end

  defp decoder_layers(
         hidden_state,
         attention_mask,
         head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         config,
         opts
       ) do
    name = opts[:name]

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, config.output_hidden_states),
      attentions: Layers.maybe_container({}, config.output_attentions),
      cross_attentions: Layers.maybe_container({}, config.output_attentions),
      cache: cache
    }

    offset = Layers.Decoder.get_cache_offset(state.cache)

    for idx <- 0..(config.decoder_layers - 1), reduce: state do
      state ->
        layer_head_mask = Axon.nx(head_mask, & &1[idx])
        cross_attention_layer_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])

        layer_cache = Layers.Decoder.get_layer_cache(state.cache, idx)

        # TODO: wrap decoder layer in a layer_drop combinator

        {hidden_state, attention, cross_attention, layer_cache} =
          decoder_layer(
            state.last_hidden_state,
            attention_mask,
            layer_head_mask,
            encoder_hidden_state,
            encoder_attention_mask,
            cross_attention_layer_head_mask,
            layer_cache,
            offset,
            config,
            name: join(name, idx)
          )

        cache = Layers.Decoder.put_layer_cache(state.cache, idx, layer_cache)

        %{
          last_hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention),
          cross_attentions: Layers.append(state.cross_attentions, cross_attention),
          cache: cache
        }
    end
  end

  defp decoder_layer(
         hidden_state,
         attention_mask,
         layer_head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_layer_head_mask,
         layer_cache,
         offset,
         config,
         opts
       ) do
    name = opts[:name]

    residual = hidden_state

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(layer_cache)

    {hidden_state, self_attention, self_attention_cache} =
      attention(
        hidden_state,
        attention_mask,
        nil,
        layer_head_mask,
        self_attention_cache,
        offset,
        config,
        num_heads: config.decoder_attention_heads,
        causal?: true,
        name: join(name, "self_attn")
      )

    hidden_state =
      hidden_state
      |> Axon.dropout(rate: config.dropout)
      |> Axon.add(residual)
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: 1.0e-5,
        name: join(name, "self_attn_layer_norm")
      )

    {hidden_state, cross_attention, cross_attention_cache} =
      Layers.if_present encoder_hidden_state do
        residual = hidden_state

        {hidden_state, cross_attention, cross_attention_cache} =
          attention(
            hidden_state,
            encoder_attention_mask,
            encoder_hidden_state,
            cross_attention_layer_head_mask,
            cross_attention_cache,
            offset,
            config,
            num_heads: config.decoder_attention_heads,
            name: join(name, "encoder_attn")
          )

        hidden_state =
          hidden_state
          |> Axon.dropout(rate: config.dropout)
          |> Axon.add(residual)
          |> Axon.layer_norm(
            channel_index: 2,
            epsilon: 1.0e-5,
            name: join(name, "encoder_attn_layer_norm")
          )

        {hidden_state, cross_attention, cross_attention_cache}
      else
        {hidden_state, Layers.none(), cross_attention_cache}
      end

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.dense(config.decoder_ffn_dim, name: join(name, "fc1"))
      |> Axon.activation(config.activation_function, name: join(name, "activation"))
      |> Axon.dropout(rate: config.activation_dropout, name: join(name, "dropout.1"))
      |> Axon.dense(config.d_model, name: join(name, "fc2"))
      |> Axon.dropout(rate: config.dropout, name: join(name, "dropout.2"))
      |> Axon.add(residual)
      |> Axon.layer_norm(channel_index: 2, epsilon: 1.0e-5, name: join(name, "final_layer_norm"))

    layer_cache =
      Layers.Decoder.put_attention_caches(
        layer_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, self_attention, cross_attention, layer_cache}
  end

  defp attention(
         hidden_state,
         attention_mask,
         cross_hidden_state,
         layer_head_mask,
         attention_cache,
         offset,
         config,
         opts
       ) do
    name = opts[:name]
    num_heads = opts[:num_heads]
    causal? = Keyword.get(opts, :causal?, false)
    cross_attention? = cross_hidden_state != nil

    query =
      hidden_state
      |> Axon.dense(config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "q_proj")
      )
      |> Layers.split_heads(num_heads)

    # For cross-attention we are given encoder hidden state
    projection_states = cross_hidden_state || hidden_state

    key =
      projection_states
      |> Axon.dense(
        config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "k_proj")
      )
      |> Layers.split_heads(num_heads)

    value =
      projection_states
      |> Axon.dense(
        config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "v_proj")
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
      |> Axon.dropout(rate: config.attention_dropout)
      |> Layers.apply_layer_head_mask(layer_head_mask)

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()
      |> Axon.dense(config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "out_proj")
      )

    {attention_output, attention_weights, attention_cache}
  end

  defp classification_head(hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dropout(rate: config.classifier_dropout)
    |> Axon.dense(config.d_model,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.activation(:tanh, name: join(name, "dense.tanh"))
    |> Axon.dropout(rate: config.classifier_dropout)
    |> Axon.dense(config.num_labels,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "out_proj")
    )
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.init_std)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["activation_function"])
      |> Shared.convert_common()
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
