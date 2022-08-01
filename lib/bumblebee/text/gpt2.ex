defmodule Bumblebee.Text.Gpt2 do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :label2id, :num_labels]

  @moduledoc """
  Models based on GPT2 architecture.

  ## Architectures

  TODO

  ## Inputs

  ## Configuration

  ### Common options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              vocab_size: 50257,
              n_position: 1024,
              n_embd: 768,
              n_layer: 24,
              n_head: 16,
              n_inner: nil,
              activation_function: :gelu_new,
              add_cross_attention: false,
              resid_pdrop: 0.1,
              embd_pdrop: 0.1,
              attn_pdrop: 0.1,
              classifier_dropout: 0.1,
              layer_norm_epsilon: 1.0e-5,
              initializer_range: 0.02,
              # Tokens
              bos_token_id: 50256,
              eos_token_id: 50256,
              pad_token_id: 50256
            ] ++ Shared.generation_defaults() ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec
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
  def base_model_prefix(), do: "transformer"

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
  def model(%__MODULE__{architecture: :for_causal_language_modeling} = config) do
    inputs = encoder_decoder_inputs(config)

    transformer_outputs = gpt2(inputs, config, name: "transformer")

    # TODO: Tie lm-head to word embedding as a config option
    logits =
      Layers.dense_transposed(transformer_outputs.last_hidden_state, config.vocab_size,
        kernel_initializer: kernel_initializer(config),
        name: "transformer.wte"
      )

    Layers.output(%{
      logits: logits,
      cache: transformer_outputs.cache,
      hidden_states: transformer_outputs.hidden_states,
      attentions: transformer_outputs.attentions,
      cross_attentions: transformer_outputs.cross_attentions
    })
  end

  @impl true
  def model(%__MODULE__{architecture: :for_token_classification} = config) do
    inputs = encoder_decoder_inputs(config)

    transformer_outputs = gpt2(inputs, config, name: "transformer")

    logits =
      transformer_outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(config))
      |> Axon.dense(config.num_labels, name: "classifier")

    Layers.output(%{
      logits: logits,
      hidden_states: transformer_outputs.hidden_states,
      attentions: transformer_outputs.attentions
    })
  end

  @impl true
  def model(%__MODULE__{architecture: :for_sequence_classification} = config) do
    inputs = encoder_decoder_inputs(config)

    transformer_outputs = gpt2(inputs, config, name: "transformer")

    logits =
      transformer_outputs.last_hidden_state
      |> Layers.dense_transposed(config.num_labels, name: "score")

    pooled_logits =
      Layers.if_present inputs["input_ids"] do
        if config.pad_token_id do
          Axon.layer(
            fn logits, input_ids, _opts ->
              {batch_size, _} = Nx.shape(input_ids)

              indices =
                input_ids
                |> Nx.not_equal(config.pad_token_id)
                |> Nx.sum(axes: [-1])
                |> Nx.subtract(1)
                |> Nx.as_type({:s, 64})

              Enum.reduce(0..(batch_size - 1), [], fn i, toks ->
                last_token_index =
                  indices
                  |> Nx.slice_along_axis(i, 1, axis: 0)
                  |> Nx.squeeze()

                last_token =
                  logits
                  |> Nx.slice_along_axis(last_token_index, 1, axis: 1)
                  |> Nx.squeeze(axes: [1])

                [last_token | toks]
              end)
              |> Enum.reverse()
              |> Nx.concatenate()
            end,
            [logits, inputs["input_ids"]]
          )
        else
          Layers.take_token(logits, axis: 1, index: -1)
        end
      else
        Layers.take_token(logits, axis: 1, index: -1)
      end

    Layers.output(%{
      logits: pooled_logits,
      cache: transformer_outputs.cache,
      hidden_states: transformer_outputs.hidden_states,
      attentions: transformer_outputs.attentions,
      cross_attentions: transformer_outputs.cross_attentions
    })
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs = encoder_decoder_inputs(config)

    inputs
    |> gpt2(config)
    |> Layers.output()
  end

  @impl true
  def init_cache(config, batch_size, max_length, inputs) do
    encoder_sequence_length =
      if encoder_last_hidden_state = inputs["encoder_last_hidden_state"] do
        Nx.axis_size(encoder_last_hidden_state, 1)
      end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: config.n_embd,
      decoder_attention_heads: config.n_head,
      encoder_attention_heads: config.n_head,
      decoder_layers: config.n_layer,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  defp gpt2(inputs, config, opts \\ []) do
    name = opts[:name]

    input_embeds =
      Layers.default inputs["input_embeds"] do
        Axon.embedding(inputs["input_ids"], config.vocab_size, config.n_embd,
          name: join(name, "wte")
        )
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(input_embeds)
      end

    position_embeds =
      Axon.embedding(position_ids, config.n_position, config.n_embd, name: join(name, "wpe"))

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(input_embeds)
      end

    decoder_attention_mask =
      Layers.default inputs["decoder_attention_mask"] do
        Layers.default_attention_mask(input_embeds)
      end

    hidden_state =
      input_embeds
      |> Axon.add(position_embeds)
      |> Axon.dropout(rate: config.embd_pdrop)

    block_outputs =
      gpt2_block_collection(
        hidden_state,
        decoder_attention_mask,
        inputs["decoder_head_mask"],
        inputs["encoder_last_hidden_state"],
        attention_mask,
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        config,
        name: join(name, "h")
      )

    last_hidden_state =
      Axon.layer_norm(block_outputs.last_hidden_state,
        channel_index: 2,
        epsilon: config.layer_norm_epsilon,
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

  defp gpt2_block_collection(
         hidden_state,
         attention_mask,
         head_mask,
         encoder_last_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         config,
         opts
       ) do
    name = opts[:name]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, config.output_hidden_states),
      attentions: Layers.maybe_container({}, config.output_attentions),
      cross_attentions: Layers.maybe_container({}, config.output_attentions),
      cache: cache
    }

    offset = Layers.Decoder.get_cache_offset(state.cache)

    outputs =
      for idx <- 0..(config.n_layer - 1), reduce: state do
        state ->
          layer_head_mask = Axon.nx(head_mask, & &1[idx])
          cross_attention_layer_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])

          layer_cache = Layers.Decoder.get_layer_cache(state.cache, idx)

          {hidden_state, attention, cross_attention, layer_cache} =
            gpt2_block(
              state.last_hidden_state,
              attention_mask,
              encoder_last_hidden_state,
              encoder_attention_mask,
              layer_head_mask,
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

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  defp gpt2_block(
         hidden_state,
         attention_mask,
         encoder_last_hidden_state,
         encoder_attention_mask,
         head_mask,
         cross_attention_head_mask,
         layer_cache,
         offset,
         config,
         opts
       ) do
    name = opts[:name]
    inner_dim = config.n_inner || 4 * config.n_embd

    residual = hidden_state

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(layer_cache)

    {attention_output, attention_weights, self_attention_cache} =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: config.layer_norm_epsilon,
        name: join(name, "ln_1")
      )
      |> gpt2_attention(
        attention_mask,
        nil,
        head_mask,
        self_attention_cache,
        offset,
        config,
        num_heads: config.n_head,
        causal?: true,
        name: join(name, "attn")
      )

    hidden_state = Axon.add(attention_output, residual)

    {hidden_state, cross_attention_weights, cross_attention_cache} =
      if config.add_cross_attention do
        Layers.if_present encoder_last_hidden_state do
          residual = hidden_state

          {cross_attention_output, cross_attention_weights, cross_attention_cache} =
            hidden_state
            |> Axon.layer_norm(
              channel_index: 2,
              epsilon: config.layer_norm_epsilon,
              name: join(name, "ln_cross_attn")
            )
            |> gpt2_attention(
              encoder_attention_mask,
              encoder_last_hidden_state,
              cross_attention_head_mask,
              cross_attention_cache,
              offset,
              config,
              name: join(name, "crossattention"),
              num_heads: config.n_head
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
        epsilon: config.layer_norm_epsilon,
        name: join(name, "ln_2")
      )
      |> gpt2_mlp(inner_dim, config, name: join(name, "mlp"))
      |> Axon.add(residual)

    layer_cache =
      Layers.Decoder.put_attention_caches(
        layer_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, attention_weights, cross_attention_weights, layer_cache}
  end

  defp gpt2_attention(
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

    {query, key, value} =
      if cross_attention? do
        q_out =
          conv1d(hidden_state, config.n_embd,
            kernel_initializer: kernel_initializer(config),
            name: join(name, "q_attn")
          )

        {query} = Axon.split(q_out, 1, axis: 1)

        kv_out =
          conv1d(hidden_state, config.n_embd * 2,
            kernel_initializer: kernel_initializer(config),
            name: join(name, "c_attn")
          )

        {key, value} = Axon.split(kv_out, 2, axis: 2)
        {query, key, value}
      else
        qkv_out =
          conv1d(hidden_state, config.n_embd * 3,
            kernel_initializer: kernel_initializer(config),
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
      |> Axon.dropout(rate: config.attn_pdrop)
      |> Layers.apply_layer_head_mask(layer_head_mask)

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()
      |> conv1d(config.n_embd,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "c_proj")
      )
      |> Axon.dropout(rate: config.resid_pdrop)

    {attention_output, attention_weights, attention_cache}
  end

  defp gpt2_mlp(hidden_state, inner_dim, config, opts) do
    name = opts[:name]

    hidden_state
    |> conv1d(inner_dim, kernel_initializer: kernel_initializer(config), name: join(name, "c_fc"))
    |> Layers.activation(config.activation_function, name: join(name, "act"))
    |> conv1d(config.n_embd,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "c_proj")
    )
    |> Axon.dropout(rate: config.resid_pdrop, name: join(name, "dropout"))
  end

  defp conv1d(input, units, opts) do
    name = opts[:name]
    kernel_initializer = opts[:kernel_initializer]
    use_bias = Keyword.get(opts, :use_bias, true)

    kernel =
      Axon.param(
        "kernel",
        fn input_shape ->
          {elem(input_shape, Nx.rank(input_shape) - 1), units}
        end,
        initializer: kernel_initializer
      )

    if use_bias do
      Axon.layer(
        fn input, kernel, bias, _opts ->
          input
          |> Nx.dot([Nx.rank(input) - 1], [], kernel, [0], [])
          |> Nx.add(bias)
        end,
        [
          input,
          kernel,
          Axon.param("bias", fn _ -> {units} end, initializer: :zeros)
        ],
        op_name: :conv1d,
        name: name
      )
    else
      Axon.layer(
        fn input, kernel, _opts ->
          Nx.dot(input, [Nx.rank(input) - 1], [], kernel, [0], [])
        end,
        [input, kernel],
        op_name: :conv1d,
        name: name
      )
    end
  end

  defp encoder_decoder_inputs(config) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, config.n_embd}
    encoder_head_mask_shape = {config.n_layer, config.n_head}
    decoder_head_mask_shape = {config.n_layer, config.n_head}

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
      Axon.input("encoder_last_hidden_state", optional: true, shape: hidden_shape),
      Axon.input("cross_attention_head_mask", optional: true, shape: decoder_head_mask_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp classifier_dropout_rate(config) do
    config.classifier_dropout || config.hidden_dropout
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.initializer_range)
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
