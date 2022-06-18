defmodule Bert do
  @moduledoc """
  BERT Model.

  Config:
    vocab_size: 30522
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    intermediate_size: 3072
    hidden_act: :gelu
    hidden_dropout_prob: 0.1
    attention_probs_dropout_prob: 0.1
    max_position_embeddings: 512
    type_vocab_size: 2
    initializer_range: 0.02
    layer_norm_eps: 1.0e-12
    pad_token_id: 0
    position_embedding_type: :absolute
    use_cache: true
    classifier_dropout: nil
  """
  # TODO: Should all be config values
  @hidden_size 768
  @vocab_size 30522
  @max_position_embedding 512
  @type_vocab_size 2
  @layer_norm_eps 1.0e-12
  @hidden_dropout_prob 0.1
  @num_attention_heads 12
  @intermediate_size 3072
  @hidden_act :gelu
  @num_hidden_layers 12

  import Bumblebee.Layers

  def bert_embeddings(input_ids, token_type_ids, position_ids, opts \\ []) do
    word_embeddings = &Axon.embedding(&1, @vocab_size, @hidden_size, kernel_initializer: :normal)

    position_embeddings =
      &Axon.embedding(&1, @max_position_embedding, @hidden_size, kernel_initializer: :normal)

    token_type_embeddings =
      &Axon.embedding(&1, @type_vocab_size, @hidden_size, kernel_initializer: :normal)

    layer_norm = &Axon.layer_norm(&1, eps: @layer_norm_eps)
    dropout = &Axon.dropout(&1, rate: @hidden_dropout_prob)

    inputs_embeds = word_embeddings.(input_ids)
    position_embeds = position_embeddings.(position_ids)
    token_type_embeds = token_type_embeddings.(token_type_ids)

    hidden_states = Axon.add([inputs_embeds, position_embeds, token_type_embeds])
    hidden_states = layer_norm.(hidden_states)
    hidden_states = dropout.(hidden_states)

    hidden_states
  end

  def bert_self_attention(hidden_states, attention_mask, opts \\ []) do
    query = &Axon.dense(&1, @hidden_size, kernel_initializer: :normal)
    key = &Axon.dense(&1, @hidden_size, kernel_initializer: :normal)
    value = &Axon.dense(&1, @hidden_size, kernel_initializer: :normal)

    head_dim = div(@hidden_size, @num_attention_heads)

    query_states = query.(hidden_states) |> Axon.reshape({:auto, @num_attention_heads, head_dim})
    value_states = value.(hidden_states) |> Axon.reshape({:auto, @num_attention_heads, head_dim})
    key_states = key.(hidden_states) |> Axon.reshape({:auto, @num_attention_heads, head_dim})

    # TODO: Missing dropout, optional bias, etc.
    # TODO: Layer head mask

    attn_bias = Axon.layer([attention_mask], &attention_bias/2, %{})

    attn_weights =
      Axon.layer([query_states, key_states, attn_bias], &dot_product_attention_weights/4, %{})

    attn_output = Axon.layer([attn_weights, value_states], &dot_product_attention_output/3, %{})
    attn_output = Axon.reshape(attn_output, {:auto, @num_attention_heads * head_dim})

    {attn_output, attn_weights}
  end

  def bert_self_output(hidden_states, input, opts \\ []) do
    dense = &Axon.dense(&1, @hidden_size, kernel_initializer: :normal)
    layer_norm = &Axon.layer_norm(&1, eps: @layer_norm_eps)
    dropout = &Axon.dropout(&1, rate: @hidden_dropout_prob)

    hidden_states = dense.(hidden_states)
    hidden_states = dropout.(hidden_states)
    hidden_states = layer_norm.(Axon.add(hidden_states, input))

    hidden_states
  end

  def bert_attention(hidden_states, attention_mask, opts \\ []) do
    self_attention = &bert_self_attention/2
    output = &bert_self_output/2

    {attn_output, attn_weights} = self_attention.(hidden_states, attention_mask)
    hidden_states = output.(attn_output, hidden_states)

    {hidden_states, attn_weights}
  end

  def bert_intermediate(hidden_states) do
    dense = &Axon.dense(&1, @intermediate_size, kernel_initializer: :normal)
    activation = &apply(Axon, @hidden_act, [&1])

    hidden_states = dense.(hidden_states)
    hidden_states = activation.(hidden_states)
    hidden_states
  end

  def bert_output(hidden_states, attention_output) do
    dense = &Axon.dense(&1, @hidden_size, kernel_initializer: :normal)
    dropout = &Axon.dropout(&1, rate: @hidden_dropout_prob)
    layer_norm = &Axon.layer_norm(&1, eps: @layer_norm_eps)

    hidden_states = dense.(hidden_states)
    hidden_states = dropout.(hidden_states)
    hidden_states = layer_norm.(Axon.add(hidden_states, attention_output))
    hidden_states
  end

  def bert_layer(hidden_states, attention_mask) do
    attention = &bert_attention/2
    intermediate = &bert_intermediate/1
    output = &bert_output/2

    {attention_outputs, attention_weights} = attention.(hidden_states, attention_mask)
    hidden_states = intermediate.(attention_outputs)
    hidden_states = output.(hidden_states, attention_outputs)

    {hidden_states, attention_weights}
  end

  def bert_layer_collection(hidden_states, attention_mask, opts \\ []) do
    {last_hidden_state, hidden_states, attentions} =
      for _ <- 1..@num_hidden_layers, reduce: {hidden_states, [], []} do
        {hidden_states, all_hidden_states, all_attention_outputs} ->
          {hidden_states, attention_weights} = bert_layer(hidden_states, attention_mask)

          {hidden_states, [hidden_states | all_hidden_states],
           [attention_weights | all_attention_outputs]}
      end

    {last_hidden_state, List.to_tuple(Enum.reverse(hidden_states)),
     List.to_tuple(Enum.reverse(attentions))}
  end

  def bert_encoder(hidden_states, attention_mask) do
    bert_layer_collection(hidden_states, attention_mask)
  end

  def bert_pooler(hidden_states) do
    dense = &Axon.dense(&1, @hidden_size, kernel_initializer: :normal)

    slice =
      &Axon.nx(&1, fn x ->
        {_, squeeze_axes} =
          x
          |> Nx.shape()
          |> Tuple.to_list()
          |> Enum.with_index()
          |> Enum.filter(fn {x, _} -> x == 1 end)
          |> Enum.unzip()

        squeeze_axes = squeeze_axes -- [0]
        Nx.slice_along_axis(x, 0, 1, axis: -1) |> Nx.squeeze(axes: squeeze_axes)
      end)

    tanh = &Axon.tanh(&1)

    cls_hidden_state = slice.(hidden_states)
    cls_hidden_state = dense.(cls_hidden_state)
    tanh.(cls_hidden_state)
  end

  def bert_module(input_ids, attention_mask, token_type_ids, position_ids, opts \\ []) do
    hidden_states = bert_embeddings(input_ids, token_type_ids, position_ids)
    {last_hidden_state, hidden_states, attentions} = bert_encoder(hidden_states, attention_mask)
    pooled = bert_pooler(last_hidden_state)

    %{
      last_hidden_state: last_hidden_state,
      pooled: pooled,
      hidden_states: hidden_states,
      attentions: attentions
    }
  end

  def model(input_shape) do
    # TODO: Config, validate, etc.
    input_ids = Axon.input(input_shape, name: "input_ids")
    attention_mask = Axon.input(input_shape, name: "attention_mask")
    token_type_ids = Axon.input(input_shape, name: "token_type_ids")
    position_ids = Axon.input(input_shape, name: "position_ids")

    bert_module(input_ids, attention_mask, token_type_ids, position_ids)
  end
end
