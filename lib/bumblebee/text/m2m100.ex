defmodule Bumblebee.Text.M2m100 do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :label2id, :num_labels]
  @moduledoc """
  TODO
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              vocab_size: 128_112,
              max_position_embeddings: 1024,
              encoder_layers: 12,
              encoder_ffn_dim: 4096,
              encoder_attention_heads: 16,
              decoder_layers: 12,
              decoder_ffn_dim: 4096,
              decoder_attention_heads: 16,
              encoder_layerdrop: 0.05,
              decoder_layerdrop: 0.05,
              activation_function: :relu,
              d_model: 1024,
              dropout: 0.1,
              attention_dropout: 0.1,
              activation_dropout: 0.0,
              init_std: 0.02,
              decoder_start_token_id: 2,
              scale_embedding: true,
              # Tokens
              pad_token_id: 1,
              bos_token_id: 0,
              eos_token_id: 2
            ] ++
              Shared.generation_defaults() ++
              Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Text.Generation

  @impl true
  def architectures(),
    do: [
      :base,
      :for_conditional_generation
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
      "input_ids" => Nx.template({1, 1}, :s64),
      "decoder_input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs = encoder_decoder_inputs(config)

    inputs
    |> m2m100(config)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_conditional_generation} = config) do
    inputs = encoder_decoder_inputs(config)
    outputs = m2m100(inputs, config, name: "model")

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

  defp m2m100(inputs, config, opts \\ []) do
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
        Axon.nx(inputs["input_ids"], fn input_ids ->
          mask = Nx.not_equal(input_ids, config.pad_token_id)

          mask
          |> Nx.cumulative_sum(axis: 1)
          |> Nx.multiply(mask)
          |> Nx.add(config.pad_token_id)
        end)
      end

    decoder_input_embeds =
      Layers.default inputs["decoder_input_embeds"] do
        token_embedding(inputs["decoder_input_ids"], config, name: join(name, "shared"))
      end
      |> Layers.or_raise(
        ~s/either "decoder_input_ids" or "decoder_inputs_embeds" must be specified/
      )

    decoder_attention_mask =
      Layers.default inputs["decoder_attention_mask"] do
        Layers.default_attention_mask(decoder_input_embeds)
      end

    decoder_position_ids =
      Layers.default inputs["decoder_position_ids"] do
        Axon.nx(inputs["decoder_input_ids"], fn input_ids ->
          mask = Nx.not_equal(input_ids, config.pad_token_id)

          mask
          |> Nx.cumulative_sum(axis: 1)
          |> Nx.multiply(mask)
          |> Nx.add(config.pad_token_id)
        end)
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

    offset = 2
    embedding_dim = config.d_model
    num_embeddings = config.max_position_embeddings + offset
    padding_idx = config.pad_token_id
    half_dim = div(embedding_dim, 2)

    position_ids
    |> Axon.nx(
      fn position_ids ->
        emb = Nx.log(10_000)
        emb = Nx.divide(emb, half_dim - 1)
        emb = Nx.exp(Nx.multiply(Nx.iota({half_dim}), Nx.negate(emb)))
        emb = Nx.multiply(Nx.new_axis(Nx.iota({num_embeddings}), 1), Nx.new_axis(emb, 0))
        emb = Nx.concatenate([Nx.sin(emb), Nx.cos(emb)], axis: 1)
        emb = Nx.reshape(emb, {num_embeddings, :auto})

        emb =
          if rem(embedding_dim, 2) == 1 do
            Nx.concatenate([emb, Nx.broadcast(0, {num_embeddings, 1})], axis: 1)
          else
            emb
          end

        zero_pad_slice = Nx.broadcast(0.0, {1, embedding_dim})
        emb = Nx.put_slice(emb, [padding_idx, 0], zero_pad_slice)

        Nx.take(emb, Nx.as_type(position_ids, {:s, 64}))
      end,
      name: join(name, "sinusoidal_position_embedding")
    )
  end

  defp encoder(input_embeds, attention_mask, position_ids, head_mask, config, opts) do
    name = opts[:name]

    position_embeds = position_embedding(position_ids, config, opts)

    encoder_outputs =
      input_embeds
      |> Axon.add(position_embeds)
      |> Axon.dropout(rate: config.dropout)
      |> encoder_layers(attention_mask, head_mask, config, name: join(name, "layers"))

    hidden_state =
      Axon.layer_norm(encoder_outputs.last_hidden_state,
        channel_index: 2,
        name: join(name, "layer_norm")
      )

    %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.append(encoder_outputs.hidden_states, hidden_state),
      attentions: encoder_outputs.attentions
    }
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
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: 1.0e-5,
        name: join(name, "self_attn_layer_norm")
      )
      |> attention(
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

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(channel_index: 2, name: join(name, "final_layer_norm"))
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
      |> Axon.dropout(rate: config.dropout, name: join(name, "dropout.2"))
      |> Axon.add(residual, name: join(name, "residual.1"))

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

    decoder_outputs =
      input_embeds
      |> Axon.add(position_embeds)
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

    hidden_state =
      decoder_outputs.last_hidden_state
      |> Axon.layer_norm(channel_index: 2, name: join(name, "layer_norm"))

    %{
      cache: Layers.Decoder.update_cache_offset(decoder_outputs.cache, input_embeds),
      last_hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions
    }
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
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: 1.0e-5,
        name: join(name, "self_attn_layer_norm")
      )
      |> attention(
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

    {hidden_state, cross_attention, cross_attention_cache} =
      Layers.if_present encoder_hidden_state do
        residual = hidden_state

        {hidden_state, cross_attention, cross_attention_cache} =
          hidden_state
          |> Axon.layer_norm(
            channel_index: 2,
            epsilon: 1.0e-5,
            name: join(name, "encoder_attn_layer_norm")
          )
          |> attention(
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

        {hidden_state, cross_attention, cross_attention_cache}
      else
        {hidden_state, Layers.none(), cross_attention_cache}
      end

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(channel_index: 2, epsilon: 1.0e-5, name: join(name, "final_layer_norm"))
      |> Axon.dense(config.decoder_ffn_dim, name: join(name, "fc1"))
      |> Axon.activation(config.activation_function, name: join(name, "activation"))
      |> Axon.dropout(rate: config.activation_dropout, name: join(name, "dropout.1"))
      |> Axon.dense(config.d_model, name: join(name, "fc2"))
      |> Axon.dropout(rate: config.dropout, name: join(name, "dropout.2"))
      |> Axon.add(residual)

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
