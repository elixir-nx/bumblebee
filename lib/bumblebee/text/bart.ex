defmodule Bumblebee.Text.Bart do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :label2id, :num_labels]

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

    * `"input_ids"` - indices of input sequence tokens in the vocabulary
      with shape `{batch_size, seq_len}`

    * `"attention_mask"` - a mask indicating which tokens to attend to.
      This is used to ignore padding tokens, which are added when
      processing a batch of sequences with different length. Defaults to
      no masked input tokens, e.g. `Nx.broadcast(1, {batch_size, seq_len})`

    * `"decoder_input_ids"` - indices of decoder input sequence tokens in
      the vocabulary with shape `{batch_size, target_seq_len}`. If not present
      and `"input_ids"` is, it will take on the value of shifting each token in
      `"input_ids"` to the right once. Otherwise it will take on a `nil` value

    * `"decoder_attention_mask"` - a mask indicating which decoder tokens to
      attend to. This is used to ignore padding tokens, which are added when
      processing a batch of sequences with different length. Defaults to
      no masked input tokens, e.g. `Nx.broadcast(1, {batch_size, target_seq_len})`

    * `"head_mask"` - a mask to nullify selected heads of the self-attention
      blocks in the encoder with shape `{encoder_layers, encoder_attention_heads}`
      Defaults to no masked heads, e.g. `Nx.broadcast(1, shape)`

    * `"decoder_head_mask"` - a mask to nullify selected heads of the
      self-attention blocks in the decoder with shape `{decoder_layers, decoder_attention_heads}`.
      Defaults to no masked heads, e.g. `Nx.broadcast(1, shape)`

    * `"cross_attn_head_mask"` - a mask to nullify selected heads of the
      cross-attention blocks in the decoder with shape `{decoder_layers, decoder_attention_heads}`
      Defaults to no masked heads, e.g. `Nx.broadcast(1, shape)`

    * `"encoder_last_hidden_state"` - last hidden state output from the
      encoder with shape `{batch_size, seq_len, hidden_size}`. Encoder last
      hidden state is used in cross-attention blocks in the decoder. If specified,
      the model will skip the encoding process and use this value directly
      for cross-attentions in the decoder

    * `"cache"` - contains pre-computed hidden states for self-attention and
      cross-attention blocks in the decoder. When `config.use_cache` is set
      to `true`, the cache will be used to speed up sequential decoding. Rather
      than recomputing hidden states for past tokens, the cache keeps a history
      of past hidden states. The cache consists of a tuple of maps, where the
      number of entries in the tuple corresponds to `decoder_layers` (e.g. one
      cache per decoder layer), and each entry is a map of tensors:
      `%{key: cached_key, value: cached_value, index: index}`. Both `cached_key`
      and `cached_value` have shapes `{batch_size, seq_len, heads, head_dim}` and
      `index` is a scalar which maintains an index of the current token in the
      sequence to decode

    * `"input_embeds"` - embedded representation of `"input_ids"` with shape
      `{batch_size, seq_len, hidden_size}` which can be specified for more
      control over how `"input_ids"` are embedded than the model's internal
      embedding lookup. If `"input_embeds"` are present, then `"input_ids"`
      will be completely ignored

    * `"decoder_input_embeds"` - embedded representation of `"decoder_input_ids"`
      with shape `{batch_size, seq_len, hidden_size}` which can be specified
      for more control over how `"decoder_input_ids"` are embedded than the model's
      internal embedding lookup. If `"decoder_input_embeds"` are present, then
      `"decoder_input_ids"` will be completely ignored

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

    * `:attention_dropout` - dropout ratio for attention probabilities.
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

    * `:use_cache` - whether or not the model should return the last key/values
      attentions. Defaults to `true`

  ### Common options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}
  """

  alias Bumblebee.Layers
  alias Bumblebee.Shared

  import Nx.Defn

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
              use_cache: true,
              pad_token_id: 1,
              bos_token_id: 0,
              eos_token_id: 2,
              is_encoder_decoder: true,
              decoder_start_token_id: 2,
              forced_eos_token_id: 2
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

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
  def model(%__MODULE__{architecture: :for_conditional_generation} = config) do
    input_shape = {nil, 11}

    outputs =
      input_shape
      |> inputs(config)
      |> bart(config)

    # TODO: Tie lm-head to word embedding as a config option
    lm_logits =
      outputs.last_hidden_state
      |> Layers.dense_transposed_layer(config.vocab_size,
        kernel_initializer: kernel_initializer(config),
        name: "shared"
      )

    Bumblebee.Utils.Model.output(
      %{
        logits: lm_logits,
        decoder_hidden_states: outputs.decoder_hidden_states,
        decoder_attentions: outputs.decoder_attentions,
        cross_attentions: outputs.cross_attentions,
        encoder_last_hidden_state: outputs.encoder_last_hidden_state,
        encoder_hidden_states: outputs.encoder_hidden_states,
        encoder_attentions: outputs.encoder_attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = config) do
    # TODO: Non-static seq len
    input_shape = {nil, 11}

    inputs = inputs(input_shape, config)
    outputs = bart(inputs, config, name: "model")

    eos_mask =
      Axon.nx(inputs["input_ids"], fn ids ->
        Nx.equal(ids, config.eos_token_id)
      end)

    sentence_representation =
      Axon.layer(
        fn eos_mask, hidden_states, _opts ->
          seq_len = Nx.axis_size(eos_mask, 1)

          eos_mask =
            eos_mask
            |> Nx.add(Nx.iota({seq_len}))
            |> Nx.multiply(1.0e-6)
            |> then(fn x ->
              max_val = x |> Nx.reduce_max(axes: [1]) |> Nx.reshape({:auto, 1})
              Nx.select(Nx.equal(x, max_val), 1, 0)
            end)

          hidden_states
          |> Nx.multiply(Nx.new_axis(eos_mask, -1))
          |> Nx.sum(axes: [1])
        end,
        [eos_mask, outputs.last_hidden_state]
      )

    logits = classification_head(sentence_representation, config, name: "classification_head")

    Bumblebee.Utils.Model.output(
      %{
        logits: logits,
        decoder_hidden_states: outputs.decoder_hidden_states,
        decoder_attentions: outputs.decoder_attentions,
        cross_attentions: outputs.cross_attentions,
        encoder_last_hidden_state: outputs.encoder_last_hidden_state,
        encoder_hidden_states: outputs.encoder_hidden_states,
        encoder_attentions: outputs.encoder_attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_question_answering} = config) do
    # TODO: Non-static seq len
    input_shape = {nil, 11}

    outputs =
      input_shape
      |> inputs(config)
      |> bart(config, name: "model")

    logits =
      Axon.dense(outputs.last_hidden_state, 2,
        kernel_initializer: kernel_initializer(config),
        name: "qa_outputs"
      )

    start_logits = Axon.nx(logits, & &1[[0..-1//1, 0..-1//1, 0]])
    end_logits = Axon.nx(logits, & &1[[0..-1//1, 0..-1//1, 1]])

    Bumblebee.Utils.Model.output(
      %{
        start_logits: start_logits,
        end_logits: end_logits,
        decoder_hidden_states: outputs.decoder_hidden_states,
        decoder_attentions: outputs.decoder_attentions,
        cross_attentions: outputs.cross_attentions,
        encoder_last_hidden_state: outputs.encoder_last_hidden_state,
        encoder_hidden_states: outputs.encoder_hidden_states,
        encoder_attentions: outputs.encoder_attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = config) do
    # TODO: Non-static seq len
    input_shape = {nil, 11}

    inputs = inputs(input_shape, config)

    input_embeds =
      Layers.maybe_layer(
        inputs["input_embeds"],
        embed_tokens(inputs["input_ids"], config, "model.decoder.embed_tokens")
      )

    outputs = decoder(inputs, input_embeds, nil, config, name: "model.decoder")

    # TODO: Tie lm-head to word embedding as a config option
    lm_logits =
      outputs.last_hidden_state
      |> Layers.dense_transposed_layer(config.vocab_size,
        kernel_initializer: kernel_initializer(config),
        name: "shared"
      )

    Bumblebee.Utils.Model.output(
      %{
        logits: lm_logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions,
        cross_attentions: outputs.cross_attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :base} = config) do
    # TODO: Non-static seq len
    input_shape = {nil, 11}

    input_shape
    |> inputs(config)
    |> bart(config)
    |> Bumblebee.Utils.Model.output(config)
  end

  # TODO: In all of the decoder_x inputs, there is a possible
  # target_seq_len that is potentially different than src_seq_len,
  # when we differentiate we also need to update the logic for the
  # default attention mask to account for this
  defp inputs({batch_size, seq_len} = input_shape, config) do
    hidden_state_shape = {batch_size, seq_len, config.d_model}

    input_ids = Axon.input(input_shape, "input_ids", default: nil)

    decoder_input_ids =
      Axon.input(input_shape, "decoder_input_ids",
        default:
          &default_decoder_input_ids(&1, config.pad_token_id, config.decoder_start_token_id)
      )

    attention_mask = Axon.input(input_shape, "attention_mask", default: &default_attention_mask/1)

    decoder_attention_mask =
      Axon.input(input_shape, "decoder_attention_mask", default: &default_attention_mask/1)

    head_mask =
      Axon.input({config.encoder_layers, config.encoder_attention_heads}, "head_mask",
        default: fn _inputs ->
          Nx.broadcast(1, {config.encoder_layers, config.encoder_attention_heads})
        end
      )

    decoder_head_mask =
      Axon.input({config.decoder_layers, config.decoder_attention_heads}, "decoder_head_mask",
        default: fn _inputs ->
          Nx.broadcast(1, {config.decoder_layers, config.decoder_attention_heads})
        end
      )

    cross_attention_head_mask =
      Axon.input(
        {config.decoder_layers, config.decoder_attention_heads},
        "cross_attention_head_mask",
        default: fn _inputs ->
          Nx.broadcast(1, {config.decoder_layers, config.decoder_attention_heads})
        end
      )

    encoder_last_hidden_state =
      Axon.input(hidden_state_shape, "encoder_last_hidden_state", default: nil)

    cache =
      Axon.input(cache_shape(input_shape, config), "cache",
        default: fn inputs ->
          head_dim = div(config.d_model, config.decoder_attention_heads)

          {batch_size, seq_len} =
            if Map.has_key?(inputs, "input_embeds") do
              {batch_size, seq_len, _} = Nx.shape(inputs["input_embeds"])
              {batch_size, seq_len}
            else
              Nx.shape(inputs["input_ids"])
            end

          single_entry_shape = {batch_size, seq_len, config.decoder_attention_heads, head_dim}
          kv_tensor = Nx.broadcast(0.0, single_entry_shape)
          index = Nx.tensor(0)
          entry = %{key: kv_tensor, value: kv_tensor, index: index}

          {entry, entry}
          |> List.duplicate(config.decoder_layers)
          |> List.to_tuple()
        end
      )

    input_embeds = Axon.input(hidden_state_shape, "input_embeds", default: nil)
    decoder_input_embeds = Axon.input(hidden_state_shape, "decoder_input_embeds", default: nil)

    %{
      "input_ids" => input_ids,
      "decoder_input_ids" => decoder_input_ids,
      "attention_mask" => attention_mask,
      "decoder_attention_mask" => decoder_attention_mask,
      "head_mask" => head_mask,
      "decoder_head_mask" => decoder_head_mask,
      "cross_attention_head_mask" => cross_attention_head_mask,
      "encoder_last_hidden_state" => encoder_last_hidden_state,
      "cache" => cache,
      "input_embeds" => input_embeds,
      "decoder_input_embeds" => decoder_input_embeds
    }
  end

  defnp default_attention_mask(inputs) do
    transform(inputs, fn inputs ->
      if Map.has_key?(inputs, "input_ids") do
        # If input_ids is present in the input map
        # then we use it's shape
        Nx.broadcast(1, inputs["input_ids"])
      else
        # Otherwise we use the embedding shape except for
        # with the hidden size
        {batch_size, seq_len, _} = Nx.shape(inputs["input_embeds"])
        Nx.broadcast(1, {batch_size, seq_len})
      end
    end)
  end

  defnp default_decoder_input_ids(inputs, pad_token_id, decoder_start_token_id) do
    transform(inputs, fn inputs ->
      if Map.has_key?(inputs, "decoder_input_embeds") do
        # If decoder_input_embeds is present in the input map
        # then we don't need decoder_input_ids at all
        nil
      else
        # If it's not then we just shift the input ids right
        # to compute the decoder input ids (e.g. the pre-training
        # task)
        inputs = inputs["input_ids"]
        batch_size = Nx.axis_size(inputs, 0)
        start_ids = Nx.broadcast(decoder_start_token_id, {batch_size, 1})
        shifted_input_ids = Nx.concatenate([start_ids, inputs[[0..-1//1, 0..-2//1]]], axis: 1)
        Nx.select(Nx.equal(shifted_input_ids, -100), pad_token_id, shifted_input_ids)
      end
    end)
  end

  defp cache_shape({batch_size, seq_len}, config) do
    head_dim = div(config.d_model, config.decoder_attention_heads)
    kv_shape = {batch_size, seq_len, config.decoder_attention_heads, head_dim}
    entry = %{key: kv_shape, value: kv_shape, index: {}}

    {entry, entry}
    |> List.duplicate(config.decoder_layers)
    |> List.to_tuple()
  end

  defp bart(inputs, config, opts \\ []) do
    name = opts[:name]

    input_embeds =
      Layers.maybe_layer(
        inputs["input_embeds"],
        embed_tokens(inputs["input_ids"], config, join(name, "shared"))
      )

    decoder_input_embeds =
      Layers.maybe_layer(
        inputs["decoder_input_embeds"],
        embed_tokens(inputs["decoder_input_ids"], config, join(name, "shared"))
      )

    encoder_outputs = encoder(inputs, input_embeds, config, name: join(name, "encoder"))

    encoder_last_hidden_state =
      Layers.maybe_layer(inputs["encoder_last_hidden_state"], encoder_outputs.last_hidden_state)

    decoder_outputs =
      decoder(inputs, decoder_input_embeds, encoder_last_hidden_state, config,
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

  defp encoder(inputs, input_embeds, config, opts) do
    name = opts[:name]

    pos_embeds = embed_positions(input_embeds, config, name)

    attention_mask = inputs["attention_mask"]
    head_mask = inputs["head_mask"]

    input_embeds
    |> Axon.add(pos_embeds)
    |> Axon.layer_norm(channel_index: 2, epsilon: 1.0e-5, name: join(name, "layernorm_embedding"))
    |> Axon.dropout(rate: config.dropout)
    |> encoder_layers(attention_mask, head_mask, config, name: join(name, "layers"))
  end

  defp embed_tokens(input_ids, config, name) do
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

  defp embed_positions(input_embeds, config, name) do
    offset = 2

    # TODO: Offset with past_key_value_length
    offset_position_ids =
      Axon.nx(input_embeds, fn embeds ->
        seq_len = Nx.axis_size(embeds, 1)
        positions = Nx.iota({seq_len})
        Nx.add(positions, offset)
      end)

    Axon.embedding(offset_position_ids, config.max_position_embeddings + offset, config.d_model,
      name: join(name, "embed_positions")
    )
  end

  defp encoder_layers(hidden_states, attention_mask, head_mask, config, opts) do
    name = opts[:name]

    initial_encoder_state = %{
      last_hidden_state: hidden_states,
      hidden_states: {hidden_states},
      attentions: {}
    }

    for idx <- 0..(config.encoder_layers - 1), reduce: initial_encoder_state do
      encoder_state ->
        layer_head_mask = Axon.nx(head_mask, & &1[idx])

        # TODO: Wrap encoder layer in a layer_drop combinator
        # that skips this connection dynamically

        {next_state, next_attention} =
          encoder_layer(encoder_state.last_hidden_state, attention_mask, layer_head_mask, config,
            name: join(name, idx)
          )

        %{
          last_hidden_state: next_state,
          hidden_states: Tuple.append(encoder_state.hidden_states, next_state),
          attentions: Tuple.append(encoder_state.attentions, next_attention)
        }
    end
  end

  defp encoder_layer(hidden_states, attention_mask, layer_head_mask, config, opts) do
    name = opts[:name]

    residual = hidden_states

    {hidden_states, attention_weights, _} =
      attention(
        hidden_states,
        attention_mask,
        nil,
        layer_head_mask,
        nil,
        config,
        num_heads: config.encoder_attention_heads,
        name: join(name, "self_attn")
      )

    hidden_states =
      hidden_states
      |> Axon.dropout(rate: config.dropout, name: join(name, "dropout.0"))
      |> Axon.add(residual, name: join(name, "residual.0"))
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: 1.0e-5,
        name: join(name, "self_attn_layer_norm")
      )

    residual = hidden_states

    hidden_states =
      hidden_states
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

    {hidden_states, attention_weights}
  end

  defp decoder(inputs, input_embeds, encoder_last_hidden_state, config, opts) do
    name = opts[:name]

    pos_embeds = embed_positions(input_embeds, config, name)

    attention_mask = inputs["decoder_attention_mask"]
    encoder_attention_mask = inputs["attention_mask"]
    head_mask = inputs["decoder_head_mask"]
    cross_attention_head_mask = inputs["cross_attention_head_mask"]
    cache = inputs["cache"]

    input_embeds
    |> Axon.add(pos_embeds)
    |> Axon.layer_norm(channel_index: 2, epsilon: 1.0e-5, name: join(name, "layernorm_embedding"))
    |> Axon.dropout(rate: config.dropout)
    |> decoder_layers(
      attention_mask,
      encoder_last_hidden_state,
      encoder_attention_mask,
      head_mask,
      cross_attention_head_mask,
      cache,
      config,
      name: join(name, "layers")
    )
  end

  defp decoder_layers(
         hidden_states,
         attention_mask,
         encoder_hidden_states,
         encoder_attention_mask,
         head_mask,
         cross_attention_head_mask,
         cache,
         config,
         opts
       ) do
    name = opts[:name]

    initial_decoder_state = %{
      last_hidden_state: hidden_states,
      hidden_states: {hidden_states},
      attentions: {},
      cross_attentions: {},
      cache: cache
    }

    for idx <- 0..(config.decoder_layers - 1), reduce: initial_decoder_state do
      decoder_state ->
        layer_head_mask = Axon.nx(head_mask, & &1[idx])
        cross_attention_layer_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])

        # TODO: Wrap entire layer in layer_drop to dynamically
        # skip layers
        {self_attention_layer_cache, cross_attention_layer_cache} =
          if config.use_cache do
            self_attention_cache = Axon.nx(decoder_state.cache, &elem(elem(&1, idx), 0))
            cross_attention_cache = Axon.nx(decoder_state.cache, &elem(elem(&1, idx), 1))
            {self_attention_cache, cross_attention_cache}
          else
            {nil, nil}
          end

        {next_state, next_attention, next_cross_attention, layer_cache} =
          decoder_layer(
            decoder_state.last_hidden_state,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            layer_head_mask,
            cross_attention_layer_head_mask,
            self_attention_layer_cache,
            cross_attention_layer_cache,
            config,
            name: join(name, idx)
          )

        updated_cache =
          if config.use_cache do
            {self_attention_cache, cross_attention_cache} = layer_cache

            Axon.layer(
              fn pkv_cache, self_attention_cache, cross_attention_cache, _opts ->
                pkv_cache
                |> put_in([Access.elem(idx), Access.elem(0)], self_attention_cache)
                |> put_in([Access.elem(idx), Access.elem(1)], cross_attention_cache)
              end,
              [decoder_state.cache, self_attention_cache, cross_attention_cache]
            )
          else
            decoder_state.cache
          end

        %{
          last_hidden_state: next_state,
          hidden_states: Tuple.append(decoder_state.hidden_states, next_state),
          attentions: Tuple.append(decoder_state.attentions, next_attention),
          cross_attentions: Tuple.append(decoder_state.cross_attentions, next_cross_attention),
          cache: updated_cache
        }
    end
  end

  defp decoder_layer(
         hidden_states,
         attention_mask,
         encoder_hidden_states,
         encoder_attention_mask,
         layer_head_mask,
         cross_attention_layer_head_mask,
         self_attention_layer_cache,
         cross_attention_layer_cache,
         config,
         opts
       ) do
    name = opts[:name]

    residual = hidden_states

    {hidden_states, self_attention_weights, self_attention_layer_cache} =
      attention(
        hidden_states,
        attention_mask,
        nil,
        layer_head_mask,
        self_attention_layer_cache,
        config,
        num_heads: config.decoder_attention_heads,
        is_decoder: true,
        is_causal: true,
        name: join(name, "self_attn")
      )

    hidden_states =
      hidden_states
      |> Axon.dropout(rate: config.dropout)
      |> Axon.add(residual)
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: 1.0e-5,
        name: join(name, "self_attn_layer_norm")
      )

    {hidden_states, cross_attention_weights, cross_attention_layer_cache} =
      if encoder_hidden_states do
        residual = hidden_states

        {hidden_states, cross_attention_weights, cross_attention_layer_cache} =
          attention(
            hidden_states,
            encoder_attention_mask,
            encoder_hidden_states,
            cross_attention_layer_head_mask,
            cross_attention_layer_cache,
            config,
            num_heads: config.decoder_attention_heads,
            is_decoder: true,
            name: join(name, "encoder_attn")
          )

        hidden_states =
          hidden_states
          |> Axon.dropout(rate: config.dropout)
          |> Axon.add(residual)
          |> Axon.layer_norm(
            channel_index: 2,
            epsilon: 1.0e-5,
            name: join(name, "encoder_attn_layer_norm")
          )

        {hidden_states, cross_attention_weights, cross_attention_layer_cache}
      else
        # TODO: Semantically these should actually be `nil`, but
        # we update the cross_attention_cache with them internally
        # so we can't just leave them blank
        {hidden_states, Axon.container({}), Axon.container({})}
      end

    residual = hidden_states

    hidden_states =
      hidden_states
      |> Axon.dense(config.decoder_ffn_dim, name: join(name, "fc1"))
      |> Axon.activation(config.activation_function, name: join(name, "activation"))
      |> Axon.dropout(rate: config.activation_dropout, name: join(name, "dropout.1"))
      |> Axon.dense(config.d_model, name: join(name, "fc2"))
      |> Axon.dropout(rate: config.dropout, name: join(name, "dropout.2"))
      |> Axon.add(residual)
      |> Axon.layer_norm(channel_index: 2, epsilon: 1.0e-5, name: join(name, "final_layer_norm"))

    layer_cache = {self_attention_layer_cache, cross_attention_layer_cache}

    {
      hidden_states,
      self_attention_weights,
      cross_attention_weights,
      layer_cache
    }
  end

  defp attention(
         hidden_states,
         attention_mask,
         key_value_states,
         layer_head_mask,
         layer_cache,
         config,
         opts
       ) do
    name = opts[:name]
    num_heads = opts[:num_heads]
    is_decoder = Keyword.get(opts, :is_decoder, false)
    is_causal = Keyword.get(opts, :is_causal, false)

    head_dim = div(config.d_model, num_heads)

    query_states =
      hidden_states
      |> Axon.dense(config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "q_proj")
      )
      |> split_heads(num_heads, head_dim)

    # For cross-attention we are given encoder hidden state
    projection_states = key_value_states || hidden_states

    key_states =
      projection_states
      |> Axon.dense(
        config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "k_proj")
      )
      |> split_heads(num_heads, head_dim)

    value_states =
      projection_states
      |> Axon.dense(
        config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "v_proj")
      )
      |> split_heads(num_heads, head_dim)

    # Prepare causal mask and combine with attention mask
    attention_mask =
      if is_causal do
        causal_mask = prepare_causal_mask(layer_cache, query_states, config)
        Axon.layer(&Layers.combine_mask/3, [attention_mask, causal_mask])
      else
        attention_mask
      end

    # If this is a decoder, then we will update the cache
    # for this layer and return the appropriate key-value
    # states, attention mask, and updated cache
    {key_states, value_states, attention_mask, layer_cache} =
      if is_decoder do
        concatenate_to_cache(query_states, key_states, value_states, attention_mask, layer_cache)
      else
        {key_states, value_states, attention_mask, layer_cache}
      end

    attention_bias = Axon.nx(attention_mask, fn x -> Nx.select(Nx.greater(x, 0), 0, -1.0e10) end)

    attention_weights =
      Axon.layer(&Layers.attention_weights/4, [query_states, key_states, attention_bias])

    attention_weights =
      Axon.dropout(attention_weights,
        rate: config.attention_dropout
      )

    attention_weights =
      Axon.layer(&Layers.apply_layer_head_mask/3, [attention_weights, layer_head_mask])

    attention_output = Axon.layer(&Layers.attention_output/3, [attention_weights, value_states])

    attention_output =
      attention_output
      |> Axon.reshape({:auto, num_heads * head_dim})
      |> Axon.dense(config.d_model,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "out_proj")
      )

    {attention_output, attention_weights, layer_cache}
  end

  defp classification_head(hidden_states, config, opts) do
    name = opts[:name]

    hidden_states
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

  defp split_heads(states, num_heads, head_dim) do
    Axon.nx(states, fn hidden ->
      shape = Nx.shape(hidden)
      new_shape = {elem(shape, 0), elem(shape, 1), num_heads, head_dim}
      Nx.reshape(hidden, new_shape)
    end)
  end

  defp prepare_causal_mask(layer_cache, query, config) do
    Axon.layer(
      fn
        %{key: cached_key, index: cached_index}, query, _opts ->
          causal_mask =
            Layers.build_causal_mask(Nx.broadcast(1, {1, config.max_position_embeddings}))

          query_length = Nx.axis_size(query, 1)
          max_decoder_length = Nx.axis_size(cached_key, 1)
          mask_shift = Nx.as_type(cached_index, {:s, 64})
          Nx.slice(causal_mask, [0, 0, mask_shift, 0], [1, 1, query_length, max_decoder_length])
      end,
      [layer_cache, query]
    )
  end

  defp concatenate_to_cache(query_states, key_states, value_states, attention_mask, layer_cache) do
    if layer_cache do
      out =
        Axon.layer(&Layers.update_cache/6, [
          query_states,
          key_states,
          value_states,
          attention_mask,
          layer_cache
        ])

      {
        Axon.nx(out, &elem(&1, 0)),
        Axon.nx(out, &elem(&1, 1)),
        Axon.nx(out, &elem(&1, 2)),
        Axon.nx(out, &elem(&1, 3))
      }
    else
      # If the cache is not initialized just return the states,
      # mask, and cache as is
      {key_states, value_states, attention_mask, layer_cache}
    end
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.init_std)
  end

  defp join(nil, rhs), do: to_string(rhs)
  defp join(lhs, rhs), do: to_string(lhs) <> "." <> to_string(rhs)

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["activation_function"])
      |> Shared.convert_common()
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
