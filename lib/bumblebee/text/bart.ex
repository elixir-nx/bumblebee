defmodule Bumblebee.Text.Bart do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :label2id, :num_labels]

  @moduledoc """
  Models based on BART architecture.

  ## Architectures

    * `:base` - plain BART without any head on top

    * `:for_causal_language_modeling` - BERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_conditional_generation` - BART with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - BART with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_question_answering` - BERT with a span classification head.
      The head returns logits for the span start and end positions

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
              use_cache: true
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
  def base_model_prefix(), do: "bart"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs({nil, 11})
    |> bart(config)
    |> Axon.container()
  end

  defp inputs(input_shape) do
    %{
      "input_ids" => Axon.input(input_shape, "input_ids"),
      # TODO: Target sequence length for decoder inputs
      "decoder_input_ids" => Axon.input(input_shape, "decoder_input_ids"),
      "attention_mask" => Axon.input(input_shape, "attention_mask"),
      "decoder_attention_mask" => Axon.input(input_shape, "decoder_attention_mask"),
      "position_ids" => Axon.input(input_shape, "position_ids"),
      "decoder_position_ids" => Axon.input(input_shape, "decoder_position_ids")
    }
  end

  defp bart(inputs, config) do
    {encoder_last_hidden_state, encoder_hidden_states, encoder_attentions} =
      encoder(inputs, config)

    {decoder_last_hidden_state, decoder_hidden_states, decoder_attentions,
     decoder_cross_attentions} = decoder(inputs, encoder_last_hidden_state, config)

    %{
      last_hidden_state: decoder_last_hidden_state,
      decoder_hidden_states: decoder_hidden_states,
      decoder_attentions: decoder_attentions,
      cross_attentions: decoder_cross_attentions,
      encoder_last_hidden_states: encoder_last_hidden_state,
      encoder_hidden_states: encoder_hidden_states,
      encoder_attentions: encoder_attentions
    }
  end

  defp encoder(inputs, config) do
    input_ids = flatten_leading(inputs["input_ids"])

    input_embeds =
      Axon.embedding(input_ids, config.vocab_size, config.d_model,
        kernel_initializer: kernel_initializer(config)
      )

    input_embeds =
      if config.scale_embedding do
        Axon.nx(input_embeds, fn x -> Nx.multiply(x, Nx.sqrt(config.d_model)) end)
      else
        input_embeds
      end

    offset_position_ids = Axon.nx(inputs["position_ids"], fn x -> Nx.add(x, 2) end)

    pos_embeds =
      Axon.embedding(offset_position_ids, config.max_position_embeddings, config.d_model)

    hidden_states = Axon.add(input_embeds, pos_embeds)
    hidden_states = Axon.layer_norm(hidden_states, epsilon: 1.0e-5)
    hidden_states = Axon.dropout(hidden_states, rate: config.dropout)

    # TODO: Perhaps this should output a map? BERT as well then?
    encoder_layer_collection(hidden_states, inputs["attention_mask"], config)
  end

  defp encoder_layer_collection(hidden_states, attention_mask, config) do
    last_hidden_state = hidden_states
    all_hidden_states = {last_hidden_state}
    all_attentions = {}

    for _idx <- 0..(config.encoder_layers - 1),
        reduce: {last_hidden_state, all_hidden_states, all_attentions} do
      {last, states, attentions} = state ->
        dropout_prob = :rand.uniform()

        if config.encoder_layerdrop >= dropout_prob do
          state
        else
          {next_state, next_attention} = encoder_layer(last, attention_mask, config)
          {next_state, Tuple.append(states, next_state), Tuple.append(attentions, next_attention)}
        end
    end
  end

  defp encoder_layer(hidden_states, attention_mask, config) do
    residual = hidden_states

    {hidden_states, self_attention_weights, _} =
      self_attention(hidden_states, attention_mask, nil, nil, config,
        num_heads: config.encoder_attention_heads
      )

    hidden_states =
      hidden_states
      |> Axon.dropout(rate: config.dropout)
      |> Axon.add(residual)
      |> Axon.layer_norm(epsilon: 1.0e-5)

    residual = hidden_states

    hidden_states =
      hidden_states
      |> Axon.dense(config.encoder_ffn_dim, kernel_initializer: kernel_initializer(config))
      |> Axon.activation(config.activation_function)
      |> Axon.dropout(rate: config.activation_dropout)
      |> Axon.dense(config.d_model, kernel_initializer: kernel_initializer(config))
      |> Axon.add(residual)
      |> Axon.layer_norm(epsilon: 1.0e-5)

    {hidden_states, self_attention_weights}
  end

  defp decoder(inputs, encoder_hidden_states, config) do
    input_ids = flatten_leading(inputs["input_ids"])

    input_embeds =
      Axon.embedding(input_ids, config.vocab_size, config.d_model,
        kernel_initializer: kernel_initializer(config)
      )

    input_embeds =
      if config.scale_embedding do
        Axon.nx(input_embeds, fn x -> Nx.multiply(x, Nx.sqrt(config.d_model)) end)
      else
        input_embeds
      end

    offset_position_ids = Axon.nx(inputs["decoder_position_ids"], fn x -> Nx.add(x, 2) end)

    pos_embeds =
      Axon.embedding(
        offset_position_ids,
        config.max_position_embeddings,
        config.d_model
      )

    hidden_states = Axon.add(input_embeds, pos_embeds)
    hidden_states = Axon.layer_norm(hidden_states, epsilon: 1.0e-5)
    hidden_states = Axon.dropout(hidden_states, rate: config.dropout)

    # TODO: Perhaps this should output a map? BERT as well then?
    decoder_layer_collection(
      hidden_states,
      inputs["decoder_attention_mask"],
      encoder_hidden_states,
      inputs["attention_mask"],
      config
    )
  end

  defp decoder_layer_collection(
         hidden_states,
         attention_mask,
         encoder_hidden_states,
         encoder_attention_mask,
         config
       ) do
    last_hidden_state = hidden_states
    past_key_values = {}
    all_hidden_states = {last_hidden_state}
    all_attentions = {}
    all_cross_attentions = {}

    for idx <- 0..(config.decoder_layers - 1),
        reduce:
          {last_hidden_state, past_key_values, all_hidden_states, all_attentions,
           all_cross_attentions} do
      {lhs, pkv, states, attentions, cross_attentions} = state ->
        # Layer drop, this randomly skips an entire layer by just forwarding
        # the current state as-is
        dropout_prob = :rand.uniform()

        if config.decoder_layerdrop >= dropout_prob do
          state
        else
          past_key_values =
            if config.use_cache do
              if tuple_size(pkv) == 0, do: nil, else: elem(pkv, idx - 1)
            else
              nil
            end

          {next_state, next_attention, next_cross_attention, past_key_value} =
            decoder_layer(
              lhs,
              attention_mask,
              encoder_hidden_states,
              encoder_attention_mask,
              past_key_values,
              config
            )

          next_cache = if config.use_cache, do: Tuple.append(pkv, past_key_value), else: pkv

          {
            next_state,
            next_cache,
            Tuple.append(states, next_state),
            Tuple.append(attentions, next_attention),
            Tuple.append(cross_attentions, next_cross_attention)
          }
        end
    end
  end

  defp decoder_layer(
         hidden_states,
         attention_mask,
         encoder_hidden_states,
         encoder_attention_mask,
         past_key_value,
         config
       ) do
    residual = hidden_states

    {past_key_value, cross_attention_past_key_value} =
      case past_key_value do
        nil ->
          {nil, nil}

        {pkv, cpkv} ->
          {pkv, cpkv}
      end

    {hidden_states, self_attention_weights, present_key_value} =
      self_attention(
        hidden_states,
        attention_mask,
        nil,
        past_key_value,
        config,
        causal: true,
        num_heads: config.decoder_attention_heads
      )

    hidden_states =
      hidden_states
      |> Axon.dropout(rate: config.dropout)
      |> Axon.add(residual)
      |> Axon.layer_norm(epsilon: 1.0e-5)

    {hidden_states, cross_attention_weights, cross_attention_present_key_value} =
      if encoder_hidden_states do
        residual = hidden_states

        {hidden_states, cross_attention_weights, cross_attention_present_key_value} =
          self_attention(
            hidden_states,
            encoder_attention_mask,
            encoder_hidden_states,
            cross_attention_past_key_value,
            config,
            num_heads: config.decoder_attention_heads
          )

        hidden_states =
          hidden_states
          |> Axon.dropout(rate: config.dropout)
          |> Axon.add(residual)
          |> Axon.layer_norm(epsilon: 1.0e-5)

        {hidden_states, cross_attention_weights, cross_attention_present_key_value}
      else
        {hidden_states, nil, nil}
      end

    residual = hidden_states

    hidden_states =
      hidden_states
      |> Axon.dense(config.decoder_ffn_dim)
      |> Axon.activation(config.activation_function)
      |> Axon.dropout(rate: config.activation_dropout)
      |> Axon.dense(config.d_model)
      |> Axon.dropout(rate: config.dropout)
      |> Axon.add(residual)
      |> Axon.layer_norm(epsilon: 1.0e-5)

    {hidden_states, self_attention_weights, cross_attention_weights,
     {present_key_value, cross_attention_present_key_value}}
  end

  defp self_attention(
         hidden_states,
         attention_mask,
         key_value_states,
         past_key_value,
         config,
         opts
       ) do
    causal = Keyword.get(opts, :causal, false)
    num_heads = Keyword.get(opts, :num_heads)

    head_dim = div(config.d_model, num_heads)

    query_states =
      Axon.dense(hidden_states, config.d_model, kernel_initializer: kernel_initializer(config))

    {key_states, value_states} =
      case {key_value_states, past_key_value} do
        {nil, nil} ->
          # If there are no past key-values and there are no given
          # key-value states, this is just a regular self attention
          # and we compute key and value states from hidden state
          key_states =
            Axon.dense(hidden_states, config.d_model,
              kernel_initializer: kernel_initializer(config)
            )

          value_states =
            Axon.dense(hidden_states, config.d_model,
              kernel_initializer: kernel_initializer(config)
            )

          {key_states, value_states}

        {key_value_states, nil} ->
          # If key-value states is present, but there are no cached
          # key-value states from previous iterations, then we will
          # recompute key-value states from the given input key-value
          # states
          key_states =
            Axon.dense(key_value_states, config.d_model,
              kernel_initializer: kernel_initializer(config)
            )

          value_states =
            Axon.dense(key_value_states, config.d_model,
              kernel_initializer: kernel_initializer(config)
            )

          {key_states, value_states}

        {nil, {past_key, past_value}} ->
          key_states =
            hidden_states
            |> Axon.dense(config.d_model, kernel_initializer: kernel_initializer(config))
            |> reshape_key_value(:auto, num_heads, head_dim)

          value_states =
            hidden_states
            |> Axon.dense(config.d_model, kernel_initializer: kernel_initializer(config))
            |> reshape_key_value(:auto, num_heads, head_dim)

          key_states = Axon.concatenate([past_key, key_states], axis: 2)
          value_states = Axon.concatenate([past_value, value_states], axis: 2)
          {key_states, value_states}
      end

    query_states = split_heads(query_states, num_heads, head_dim)
    key_states = split_heads(key_states, num_heads, head_dim)
    value_states = split_heads(value_states, num_heads, head_dim)

    # TODO: Causal mask

    attention_bias = Axon.layer(&Layers.attention_bias/2, [attention_mask])

    attention_weights =
      Axon.layer(&Layers.attention_weights/4, [query_states, key_states, attention_bias])

    attention_weights =
      Axon.dropout(attention_weights,
        rate: config.attention_dropout
      )

    attention_output = Axon.layer(&Layers.attention_output/3, [attention_weights, value_states])

    attention_output =
      attention_output
      |> Axon.reshape({:auto, num_heads * head_dim})
      |> Axon.dense(config.d_model, kernel_initializer: kernel_initializer(config))

    {attention_output, attention_weights, {key_states, value_states}}
  end

  defp flatten_leading(%Axon{} = x) do
    Axon.nx(x, fn x ->
      shape =
        x
        |> Nx.shape()
        |> Tuple.delete_at(0)
        |> put_elem(0, :auto)

      Nx.reshape(x, shape)
    end)
  end

  defp split_heads(states, num_heads, head_dim) do
    Axon.nx(states, fn hidden ->
      shape = Nx.shape(hidden)
      new_shape = {elem(shape, 0), elem(shape, 1), num_heads, head_dim}
      Nx.reshape(hidden, new_shape)
    end)
  end

  defp reshape_key_value(key_value, seq_len, num_heads, head_dim) do
    Axon.nx(key_value, fn kv ->
      shape = Nx.shape(kv)
      new_shape = {elem(shape, 0), seq_len, num_heads, head_dim}
      kv |> Nx.reshape(new_shape) |> Nx.transpose(axes: [0, 2, 1, 3])
    end)
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.init_std)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.atomize_values(["activation_function"])
      |> Shared.cast_common_values()
      |> Shared.data_into_config(config)
    end
  end
end
