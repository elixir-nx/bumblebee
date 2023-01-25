defmodule Bumblebee.Audio.Whisper do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 51865,
        doc: """
        the vocabulary size of the model. This corresponds to the number of distinct
        tokens that can be represented by the decoder
        """
      ],
      num_mel_bins: [
        default: 80,
        doc: """
        the number of mel features used per input features
        """
      ],
      encoder_max_positions: [
        default: 1500,
        doc: """
        the vocabulary size of the encoder position embedding. This corresponds to the maximum
        sequence length of log-mel filter-bank features that the model can process
        """
      ],
      decoder_max_positions: [
        default: 448,
        doc: """
        the vocabulary size of the decoder position embedding. This corresponds to the maximum
        sequence length that this model can generate. Typically this is set to a large value just
        in case, such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      encoder_num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      decoder_num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the decoder"
      ],
      encoder_num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      decoder_num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      encoder_intermediate_size: [
        default: 4096,
        docs:
          "the dimensionality of the intermediate (often named feed-forward) layer in the encoder"
      ],
      decoder_intermediate_size: [
        default: 4096,
        docs:
          "the dimensionality of the intermediate (often named feed-forward) layer in the decoder"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for encoder and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      activation_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for activations inside fully connected layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions
      ]) ++
      Shared.token_options(
        pad_token_id: 50256,
        bos_token_id: 50257,
        eos_token_id: 50256,
        decoder_start_token_id: 50257
      ) ++ Shared.generation_options(forced_bos_token_id: 0, forced_eos_token_id: 2)

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_conditional_generation]

  @impl true
  def config(spec, opts \\ []) do
    spec
    |> Shared.put_config_attrs(opts)
  end

  @impl true
  def input_template(spec) do
    input_features_length = 2 * spec.encoder_max_positions

    %{
      "input_features" => Nx.template({1, spec.num_mel_bins, input_features_length}, :s64),
      "decoder_input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs(spec)
    |> whisper(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_conditional_generation} = spec) do
    inputs = inputs(spec)
    outputs = whisper(inputs, spec, name: "model")

    lm_logits =
      outputs.hidden_state
      |> Layers.dense_transposed(spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: "model.decoder.embed_tokens"
      )

    Layers.output(%{
      logits: lm_logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_hidden_state: outputs.encoder_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions,
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
      decoder_num_attention_heads: spec.decoder_num_attention_heads,
      encoder_num_attention_heads: spec.encoder_num_attention_heads,
      decoder_num_blocks: spec.decoder_num_blocks,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  defp inputs(spec) do
    input_features_length = 2 * spec.encoder_max_positions

    encoder_input_shape = {nil, spec.num_mel_bins, input_features_length}
    decoder_input_shape = {nil, nil}

    encoder_attention_head_mask_shape =
      {spec.encoder_num_blocks, spec.encoder_num_attention_heads}

    decoder_attention_head_mask_shape =
      {spec.decoder_num_blocks, spec.decoder_num_attention_heads}

    hidden_shape = {nil, nil, spec.hidden_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_features", shape: encoder_input_shape),
      Axon.input("attention_head_mask", optional: true, shape: encoder_attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("decoder_input_ids", optional: true, shape: decoder_input_shape),
      Axon.input("decoder_attention_mask", optional: true, shape: decoder_input_shape),
      Axon.input("decoder_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("decoder_input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
      Axon.input("cross_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("cache", optional: true)
    ])
  end

  defp whisper(inputs, spec, opts \\ []) do
    name = opts[:name]

    input_embeddings =
      Layers.default inputs["input_embeddings"] do
        feature_embedding(inputs["input_features"], spec, name: join(name, "encoder"))
      end

    decoder_input_embeddings =
      Layers.default inputs["decoder_input_embeddings"] do
        decoder_input_ids =
          Layers.default inputs["decoder_input_ids"] do
            Layers.shift_tokens_right(inputs["decoder_input_ids"], spec.decoder_start_token_id)
          end

        Axon.embedding(decoder_input_ids, spec.vocab_size, spec.hidden_size,
          name: join(name, "decoder.embed_tokens")
        )
      end

    decoder_attention_mask =
      Layers.default inputs["decoder_attention_mask"] do
        Layers.default_attention_mask(decoder_input_embeddings)
      end

    encoder_outputs =
      Layers.if_present inputs["encoder_hidden_state"] do
        %{
          hidden_state: inputs["encoder_hidden_state"],
          hidden_states: Layers.none(),
          attentions: Layers.none()
        }
      else
        encoder(
          input_embeddings,
          inputs["attention_head_mask"],
          spec,
          name: join(name, "encoder")
        )
      end

    # whisper does not support masking input features, but decoder
    # needs an attention mask for the encoder
    default_attention_mask = Axon.constant(Nx.tensor([[1]]))

    decoder_outputs =
      decoder(
        decoder_input_ids,
        decoder_input_embeddings,
        decoder_attention_mask,
        inputs["decoder_attention_head_mask"],
        encoder_outputs.hidden_state,
        default_attention_mask,
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        name: join(name, "decoder")
      )

    %{
      hidden_state: decoder_outputs.hidden_state,
      decoder_hidden_states: decoder_outputs.hidden_states,
      decoder_attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions,
      cache: decoder_outputs.cache,
      encoder_hidden_state: encoder_outputs.hidden_state,
      encoder_hidden_states: encoder_outputs.hidden_states,
      encoder_attentions: encoder_outputs.attentions
    }
  end

  defp encoder(
         input_embeddings,
         attention_head_mask,
         spec,
         opts
       ) do
    name = opts[:name]

    position_embeddings = feature_position_embedding(spec, name: join(name, "embed_positions"))

    encoder_outputs =
      input_embeddings
      |> Axon.transpose([0, 2, 1], name: join(name, "inputs.permute"))
      |> Axon.add(position_embeddings)
      |> Axon.dropout(rate: spec.dropout_rate)
      |> encoder_blocks(attention_head_mask, spec, name: join(name, "layers"))

    hidden_state = Axon.layer_norm(encoder_outputs.hidden_state, name: join(name, "layer_norm"))

    %{
      encoder_outputs
      | hidden_state: hidden_state,
        hidden_states: Layers.append(encoder_outputs.hidden_states, hidden_state)
    }
  end

  defp feature_embedding(input_features, spec, opts) do
    name = opts[:name]

    input_features
    |> Axon.conv(spec.hidden_size,
      kernel_size: 3,
      padding: [{1, 1}],
      channels: :first,
      name: join(name, "conv1")
    )
    |> Axon.gelu()
    |> Axon.conv(spec.hidden_size,
      kernel_size: 3,
      strides: [2],
      padding: [{1, 1}],
      name: join(name, "conv2"),
      channels: :first
    )
    |> Axon.gelu()
  end

  defp feature_position_embedding(spec, opts) do
    name = opts[:name]

    # strangely, they just use this randomly initialized embedding
    # weight as the position embedding
    kernel =
      Axon.param("weight", fn ->
        Axon.Shape.embedding_kernel({}, spec.encoder_max_positions, spec.hidden_size)
      end)

    Axon.layer(fn kernel, _opts -> kernel end, [kernel], name: name)
  end

  defp encoder_blocks(hidden_state, attention_head_mask, spec, opts) do
    name = opts[:name]

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      attentions: Layers.maybe_container({}, spec.output_attentions)
    }

    for idx <- 0..(spec.encoder_num_blocks - 1), reduce: state do
      state ->
        block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])

        # TODO: wrap encoder block in a layer_drop combinator

        {hidden_state, attention} =
          encoder_block(state.hidden_state, block_attention_head_mask, spec, name: join(name, idx))

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention)
        }
    end
  end

  defp encoder_block(hidden_state, block_attention_head_mask, spec, opts) do
    name = opts[:name]

    default_attention_mask = Axon.constant(Nx.tensor([[1]]))

    residual = hidden_state

    {hidden_state, attention_weights, _} =
      hidden_state
      |> Axon.layer_norm(name: join(name, "self_attn_layer_norm"))
      |> attention(
        default_attention_mask,
        nil,
        block_attention_head_mask,
        Layers.none(),
        Layers.none(),
        spec,
        num_heads: spec.encoder_num_attention_heads,
        name: join(name, "self_attn")
      )

    hidden_state =
      hidden_state
      |> Axon.dropout(rate: spec.dropout_rate)
      |> Axon.add(residual)

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(name: join(name, "final_layer_norm"))
      |> Axon.dense(spec.encoder_intermediate_size, name: join(name, "fc1"))
      |> Layers.activation(spec.activation, name: join(name, "fc1.activation"))
      |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "fc1.dropout"))
      |> Axon.dense(spec.hidden_size, name: join(name, "fc2"))
      |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "fc2.dropout"))
      |> Axon.add(residual)

    {hidden_state, attention_weights}
  end

  defp decoder(
         decoder_input_ids,
         input_embeddings,
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

    position_embeddings =
      token_position_embedding(decoder_input_ids, cache, spec, name: join(name, "embed_positions"))

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)

    outputs =
      input_embeddings
      |> Axon.add(position_embeddings)
      |> Axon.dropout(rate: spec.dropout_rate)
      |> decoder_blocks(
        attention_mask,
        attention_head_mask,
        encoder_hidden_state,
        encoder_attention_mask,
        cross_attention_head_mask,
        cache,
        spec,
        name: join(name, "layers")
      )

    hidden_state = Axon.layer_norm(outputs.hidden_state, name: join(name, "layer_norm"))

    outputs = %{
      outputs
      | hidden_state: hidden_state,
        hidden_states: Layers.append(outputs.hidden_states, hidden_state)
    }

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, input_embeddings))
  end

  defp token_position_embedding(input_ids, cache, spec, opts) do
    name = opts[:name]

    # Again this is just an embedding weight, but they
    # offset this by cache size
    kernel =
      Axon.param("weight", fn _, _ ->
        Axon.Shape.embedding_kernel({}, spec.decoder_max_positions, spec.hidden_size)
      end)

    offset = Layers.Decoder.get_cache_offset(cache)

    Axon.layer(
      fn input_ids, kernel, offset, _opts ->
        offset =
          case offset do
            %Axon.None{} -> 0
            offset -> Nx.as_type(offset, {:s, 64})
          end

        input_sequence_length = elem(Nx.shape(input_ids), 1)
        Nx.slice_along_axis(kernel, offset, input_sequence_length)
      end,
      [input_ids, kernel, Axon.optional(offset)],
      name: name
    )
  end

  defp decoder_blocks(
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

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      attentions: Layers.maybe_container({}, spec.output_attentions),
      cross_attentions: Layers.maybe_container({}, spec.output_attentions),
      cache: cache
    }

    offset = Layers.Decoder.get_cache_offset(state.cache)

    for idx <- 0..(spec.decoder_num_blocks - 1), reduce: state do
      state ->
        block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])
        cross_attention_block_attention_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])

        block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

        # TODO: wrap decoder block in a layer_drop combinator

        {hidden_state, attention, cross_attention, block_cache} =
          decoder_block(
            state.hidden_state,
            attention_mask,
            block_attention_head_mask,
            encoder_hidden_state,
            encoder_attention_mask,
            cross_attention_block_attention_head_mask,
            block_cache,
            offset,
            spec,
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

  defp decoder_block(
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

    residual = hidden_state

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    {hidden_state, self_attention, self_attention_cache} =
      hidden_state
      |> Axon.layer_norm(name: join(name, "self_attn_layer_norm"))
      |> attention(
        attention_mask,
        nil,
        block_attention_head_mask,
        self_attention_cache,
        offset,
        spec,
        num_heads: spec.decoder_num_attention_heads,
        causal?: true,
        name: join(name, "self_attn")
      )

    hidden_state =
      hidden_state
      |> Axon.dropout(rate: spec.dropout_rate)
      |> Axon.add(residual)

    {hidden_state, cross_attention, cross_attention_cache} =
      Layers.if_present encoder_hidden_state do
        residual = hidden_state

        {hidden_state, cross_attention, cross_attention_cache} =
          hidden_state
          |> Axon.layer_norm(name: join(name, "encoder_attn_layer_norm"))
          |> attention(
            encoder_attention_mask,
            encoder_hidden_state,
            cross_attention_block_attention_head_mask,
            cross_attention_cache,
            offset,
            spec,
            num_heads: spec.decoder_num_attention_heads,
            name: join(name, "encoder_attn")
          )

        hidden_state =
          hidden_state
          |> Axon.dropout(rate: spec.dropout_rate)
          |> Axon.add(residual)

        {hidden_state, cross_attention, cross_attention_cache}
      else
        {hidden_state, Layers.none(), cross_attention_cache}
      end

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(name: join(name, "final_layer_norm"))
      |> Axon.dense(spec.decoder_intermediate_size, name: join(name, "fc1"))
      |> Axon.activation(spec.activation, name: join(name, "activation"))
      |> Axon.dropout(rate: spec.activation_dropout_rate, name: join(name, "dropout.1"))
      |> Axon.dense(spec.hidden_size, name: join(name, "fc2"))
      |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout.2"))
      |> Axon.add(residual)

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, self_attention, cross_attention, block_cache}
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
    num_heads = opts[:num_heads]
    causal? = Keyword.get(opts, :causal?, false)
    cross_attention? = cross_hidden_state != nil

    query =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "q_proj")
      )
      |> Layers.split_heads(num_heads)

    # For cross-attention we are given encoder hidden state
    projection_states = cross_hidden_state || hidden_state

    key =
      projection_states
      |> Axon.dense(
        spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "k_proj"),
        use_bias: false
      )
      |> Layers.split_heads(num_heads)

    value =
      projection_states
      |> Axon.dense(
        spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
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
      |> Axon.dropout(rate: spec.attention_dropout_rate)
      |> Layers.apply_attention_head_mask(block_attention_head_mask)

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "out_proj")
      )

    {attention_output, attention_weights, attention_cache}
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
          hidden_size: {"d_model", number()},
          num_mel_bins: {"num_mel_bins", number()},
          encoder_max_positions: {"max_source_positions", number()},
          decoder_max_positions: {"max_target_positions", number()},
          encoder_num_blocks: {"encoder_layers", number()},
          decoder_num_blocks: {"decoder_layers", number()},
          encoder_num_attention_heads: {"encoder_attention_heads", number()},
          decoder_num_attention_heads: {"decoder_attention_heads", number()},
          encoder_intermediate_size: {"encoder_ffn_dim", number()},
          decoder_intermediate_size: {"decoder_ffn_dim", number()},
          activation: {"activation_function", atom()},
          dropout_rate: {"dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          activation_dropout_rate: {"activation_dropout", number()},
          initializer_scale: {"init_std", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
