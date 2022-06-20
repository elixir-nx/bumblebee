defmodule Bumblebee.Vision.Beit do
  @common_keys [:id2label, :label2id, :num_labels, :output_hidden_states]

  @moduledoc """
  Models based on the BEiT architecture.

  ## Architectures

    * `:base` - plain BEiT without any head on top.

    * `:for_image_classification` - BEiT with a classification head.
      The head consists of a single dense layer on top of the pooled
      features

    * `:for_masked_image_modeling` - BEiT with a language modeling
      head on top for predicting visual tokens.

  ## Config

    * `:vocab_size` - vocab size of the model. Number of different image
      tokens to use during pre-training. Defaults to `8192`

    * `:hidden_size` - dimensionality of the encoder layers and the pooler
      layer. Defaults to `768`

    * `:num_hidden_layers` - number of hidden layers in the transformer
      encoder. Defaults to `12`

    * `:num_attention_heads` - number of attention heads for each attention
      layer in the transformer encoder. Defaults to `12`

    * `:intermediate_size` - dimensionality of the intermediate layer in
      the transformer encoder. Defaults to `3072`

    * `:hidden_act` - non-linear activation function in the encoder and
      pooler. Defaults to `:gelu`

    * `:hidden_dropout_prob` - dropout probability for all fully connected
      layers in the embeddings, encoder, and pooler. Defaults to `0.0`

    * `:attention_probs_dropout_prob` - dropout ratio for attention
      probabilities. Defaults to `0.0`

    * `:initializer_range` - standard deviation of the truncated normal
      initializer for initializing all weight matrices. Defaults to `0.02`

    * `:layer_norm_eps` - epsilon used for layer norm layers. Defaults to
      `1.0e-12`

    * `:image_size` - resolution of each image. Defaults to `224`

    * `:patch_size` - resolution of each patch. Defaults to `16`

    * `:num_channels` - number of input channels. Defaults to `3`

    * `:use_mask_token` - whether or not to use a mask token for masked
      image modeling. Defaults to `false`

    * `:use_absolute_position_embeddings` - whether to use BERT-style absolute
      position embeddings. Defaults to `false`

    * `:use_relative_position_bias` - whether to use T5-style relative position
      embeddings in self-attention layers. Defaults to `false`

    * `:use_shared_relative_position_bias` - whether to use the same relative
      position embeddings across all self-attention layers of the transformer.
      Defaults to `false`

    * `:layer_scale_init_value` - scale to use in the self-attention layers.
      Defaults to `0.1`

    * `:drop_path_rate` - stochastic depth rate per sample. Defaults to `0.1`

    * `:use_mean_pooling` - Whether to mean-pool the final hidden states of the
      patches instead of using the final hidden state of the CLS token. Defaults
      to `true`

    * `:out_indices` - indices of feature maps to use for semantic segmentation.
      Defaults to `[3, 5, 7, 11]`

    * `:use_auxiliary_head` - whether to use an auxiliary head during training.
      Defaults to `true`

    * `:auxiliary_loss_weight` - weight of cross-entropy loss of the auxiliary
      head. Defaults to `0.4`

    * `:auxiliary_channels` - number of channels to use in auxiliary head. Defaults
      to `256`

    * `:auxiliary_num_convs` - number of convolutional layers to use in the
      auxiliary head. Defaults to `256`

    * `:auxiliary_concat_input` - whether to concatenate the output of the auxiliary
      head with the input before the classification layer. Defaults to `false`

    * `:semantic_loss_ignore_index` - index that is ignored by the loss function of
      the semantic segmentation model. Defaults to `255`

  ### Common options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}
  """

  alias Bumblebee.Layers
  alias Bumblebee.Shared

  defstruct [
              architecture: :base,
              vocab_size: 8192,
              hidden_size: 768,
              num_hidden_layers: 12,
              num_attention_heads: 12,
              intermediate_size: 3072,
              hidden_act: :gelu,
              hidden_dropout_prob: 0.0,
              attention_probs_dropout_prob: 0.0,
              initializer_range: 0.02,
              layer_norm_eps: 1.0e-12,
              image_size: 224,
              patch_size: 16,
              num_channels: 3,
              use_mask_token: false,
              use_absolute_position_embedding: false,
              use_relative_position_bias: false,
              use_shared_relative_position_bias: false,
              layer_scale_init_value: 0.1,
              drop_path_rate: 0.1,
              use_mean_pooling: true,
              out_indices: [3, 5, 7, 11],
              pool_scales: [1, 2, 3, 6],
              use_auxiliary_head: true,
              auxiliary_loss_weight: 0.4,
              auxiliary_channels: 256,
              auxiliary_num_convs: 1,
              auxiliary_concat_input: false,
              semantic_loss_ignore_index: 255
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

  @impl true
  def architectures(),
    do: [:base, :for_image_classification, :for_masked_image_modeling]

  @impl true
  def base_model_prefix(), do: "beit"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def model(%__MODULE__{architecture: :for_image_classification} = config) do
    outputs = base_model(config, "beit")

    pooled_output = outputs.pooler_output
    logits = Axon.dense(pooled_output, config.num_labels, name: "classifier")

    Axon.container(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_masked_image_modeling} = config) do
    # TODO: This model specifically has additional inputs, namely the bool_masked_pos
    # and head_mask inputs, so we need to reflect that when building the base model,
    # this goes along with some of previous issues with reflecting optional inputs
    # in an Axon model
    outputs = base_model(config, "beit")

    prediction_scores =
      outputs.last_hidden_state
      |> Axon.layer_norm(channel_index: 2, epsilon: config.layer_norm_eps, name: "layernorm")
      |> Axon.nx(& &1[[0..-1//1, 1..-1//1, 0..-1//1]])
      |> Axon.dense(config.vocab_size, name: "lm_head")

    Axon.container(%{
      logits: prediction_scores,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  # TODO: See PyTorch implementation, the head on this one ends up
  # getting rather complex, so I'm leaving as a TODO for now
  # def model(%__MODULE__{architecture: :for_semantic_segmentation} = config) do
  # end

  def model(%__MODULE__{architecture: :base} = config) do
    config
    |> base_model("beit")
    |> Axon.container()
  end

  defp base_model(config, name) do
    input_shape = {nil, config.num_channels, config.image_size, config.image_size}
    pixel_values = Axon.input(input_shape, "pixel_values")

    hidden_states = embedding(pixel_values, config, name: join(name, "embeddings"))

    {last_hidden_state, all_hidden_states, all_attentions} =
      encoder(hidden_states, config, name: join(name, "encoder"))

    pooled = pooler(last_hidden_state, config, name: join(name, "pooler"))

    %{
      last_hidden_state: last_hidden_state,
      pooler_output: pooled,
      hidden_states: all_hidden_states,
      attentions: all_attentions
    }
  end

  defp embedding(pixel_values, config, opts) do
    name = opts[:name]

    pixel_values
    |> patch_embeddings(config, name: join(name, "patch_embeddings"))
    |> maybe_mask_pos(config)
    |> concat_cls_token(config, name: name)
    |> maybe_position_bias(config, name: name)
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
  end

  defp patch_embeddings(pixel_values, config, opts) do
    name = opts[:name]

    embeddings =
      Axon.conv(pixel_values, config.hidden_size,
        kernel_size: config.patch_size,
        strides: config.patch_size,
        padding: :valid,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "projection")
      )

    {_, channels, h, w} = shape(embeddings)

    embeddings
    |> Axon.reshape({channels, h * w},
      name: join(name, "reshape")
    )
    |> Axon.transpose([0, 2, 1], ignore_batch?: false, name: join(name, "transpose"))
  end

  defp maybe_mask_pos(embeds, config) do
    mask_token = Axon.param("mask_token", {1, 1, config.hidden_size}, initializer: :zeros)

    # TODO: This is only applied if `bool_masked_pos` is provided
    # as an input? It also seems like there is a bug here in the
    # torch implementation. Notice the new axis adds a trailing 1
    # to w and thus it is not broadcastable to the embeddings
    if false do
      Axon.layer(
        fn token, embed, _opts ->
          {batch_size, seq_len, _} = Nx.shape(embed)
          token = Nx.broadcast(token, {batch_size, seq_len, config.hidden_size})
          w = Nx.new_axis(token, -1)

          embed
          |> Nx.multiply(Nx.subtract(1, w))
          |> Nx.add(Nx.multiply(token, w))
        end,
        [mask_token, embeds]
      )
    else
      embeds
    end
  end

  defp maybe_position_bias(embeds, config, opts) do
    name = opts[:name]

    num_patches =
      div(config.image_size, config.patch_size) * div(config.image_size, config.patch_size)

    position_embeddings =
      Axon.param("position_embeddings", {1, num_patches + 1, config.hidden_size},
        initializer: :zeros
      )

    if config.use_absolute_position_embedding do
      Axon.layer(fn pos, embed, _opts -> Nx.add(pos, embed) end, [position_embeddings, embeds],
        name: name
      )
    else
      embeds
    end
  end

  defp concat_cls_token(embeds, config, opts) do
    name = opts[:name]

    cls_token = Axon.param("cls_token", {1, 1, config.hidden_size}, initializer: :zeros)

    Axon.layer(
      fn token, embed, _opts ->
        batch_size = Nx.axis_size(embed, 0)
        token = Nx.broadcast(token, {batch_size, 1, config.hidden_size})
        token = Nx.as_type(token, Nx.type(embed))
        Nx.concatenate([token, embed], axis: 1)
      end,
      [cls_token, embeds],
      name: name
    )
  end

  defp encoder(hidden_states, config, opts) do
    name = opts[:name]

    window_size =
      {div(config.image_size, config.patch_size), div(config.image_size, config.patch_size)}

    drop_path_rates =
      Nx.to_flat_list(linspace(0, config.drop_path_rate, config.num_hidden_layers))

    shared_relative_position_bias =
      if config.use_shared_relative_position_bias do
        relative_position_bias(config,
          window_size: window_size,
          name: join(name, "relative_position_bias")
        )
      else
        nil
      end

    beit_layer_collection(hidden_states, shared_relative_position_bias, config,
      drop_path_rates: drop_path_rates,
      name: name
    )
  end

  defp beit_layer_collection(hidden_states, shared_relative_position_bias, config, opts) do
    name = opts[:name]
    drop_path_rates = opts[:drop_path_rates]

    last_hidden_state = hidden_states
    all_hidden_states = {last_hidden_state}
    all_attentions = {}

    for idx <- 0..(config.num_hidden_layers - 1),
        reduce: {last_hidden_state, all_hidden_states, all_attentions} do
      {lhs, states, attns} ->
        layer_name = join("layer", "#{idx}")

        {state, attention} =
          beit_layer(lhs, shared_relative_position_bias, config,
            drop_path_rate: Enum.at(drop_path_rates, idx),
            name: join(name, layer_name)
          )

        {state, Tuple.append(states, state), Tuple.append(attns, attention)}
    end
  end

  defp beit_layer(hidden_states, shared_relative_position_bias, config, opts) do
    name = opts[:name]

    drop_path_rate = opts[:drop_path_rate]

    normalized_states =
      Axon.layer_norm(hidden_states,
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "layernorm_before")
      )

    {attention_output, attentions} =
      attention(normalized_states, shared_relative_position_bias, config,
        name: join(name, "attention")
      )

    attention_output =
      if config.layer_scale_init_value > 0 do
        Layers.scale_layer(attention_output,
          name: name,
          scale_name: "lambda_1",
          scale_init_value: config.layer_scale_init_value,
          channel_index: 2
        )
      else
        attention_output
      end

    hidden_states =
      attention_output
      |> Layers.drop_path_layer(rate: drop_path_rate, name: join(name, "drop_path.0"))
      |> Axon.add(hidden_states, name: join(name, "residual.0"))

    layer_output =
      hidden_states
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "layernorm_after")
      )
      |> intermediate(config, name: join(name, "intermediate"))
      |> output(config, name: join(name, "output"))

    output =
      if config.layer_scale_init_value > 0 do
        Layers.scale_layer(layer_output,
          name: name,
          scale_name: "lambda_2",
          scale_init_value: config.layer_scale_init_value,
          channel_index: 2
        )
      else
        layer_output
      end

    output =
      output
      |> Layers.drop_path_layer(rate: drop_path_rate, name: join(name, "drop_path.1"))
      |> Axon.add(hidden_states, name: join(name, "residual.1"))

    {output, attentions}
  end

  defp attention(hidden_states, shared_relative_position_bias, config, opts) do
    name = opts[:name]

    {attention_output, attentions} =
      self_attention(hidden_states, shared_relative_position_bias, config,
        name: join(name, "attention")
      )

    attention_output = self_output(attention_output, config, name: join(name, "output"))

    {attention_output, attentions}
  end

  defp self_attention(hidden_states, shared_relative_position_bias, config, opts) do
    name = opts[:name]

    window_size =
      {div(config.image_size, config.patch_size), div(config.image_size, config.patch_size)}

    {_, seq_len, _} = shape(hidden_states)
    head_dim = div(config.hidden_size, config.num_attention_heads)
    out_shape = {seq_len, config.num_attention_heads, head_dim}

    query_states =
      Axon.dense(hidden_states, config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "query")
      )

    key_states =
      Axon.dense(hidden_states, config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "key"),
        use_bias: false
      )

    value_states =
      Axon.dense(hidden_states, config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "value")
      )

    query_states = Axon.reshape(query_states, out_shape)
    key_states = Axon.reshape(key_states, out_shape)
    value_states = Axon.reshape(value_states, out_shape)

    attention_bias =
      if config.use_relative_position_bias do
        attention_bias =
          relative_position_bias(config,
            window_size: window_size,
            name: join(name, "relative_position_bias")
          )

        Axon.nx(attention_bias, &Nx.new_axis(&1, 0))
      else
        Axon.constant(Nx.tensor(0.0))
      end

    attention_bias =
      if shared_relative_position_bias do
        Axon.add(attention_bias, shared_relative_position_bias)
      else
        attention_bias
      end

    attention_weights =
      Axon.layer(&Layers.attention_weights/4, [query_states, key_states, attention_bias])

    attention_weights = Axon.dropout(attention_weights, rate: config.attention_probs_dropout_prob)

    attention_output = Axon.layer(&Layers.attention_output/3, [attention_weights, value_states])

    attention_output =
      Axon.reshape(attention_output, {:auto, config.num_attention_heads * head_dim})

    {attention_output, attention_weights}
  end

  defp relative_position_bias(config, opts) do
    name = opts[:name]
    {h, w} = opts[:window_size]

    num_relative_distance = (2 * h - 1) * (2 * w - 1) + 3

    relative_position_bias_table =
      Axon.param(
        "relative_position_bias_table",
        {num_relative_distance, config.num_attention_heads},
        initializer: :zeros
      )

    # TODO: Correct the initializer according to PyTorch and Flax:
    # https://github.com/huggingface/transformers/blob/6589e510fa4e6c442059de2fab84752535de9b23/src/transformers/models/beit/modeling_flax_beit.py#L119
    # Interestingly in PyTorch this is a "buffer" and in Flax this
    # is a staic array. I'm curious if we are supposed to differentiate
    # this as just a static variable or something. For now we can
    # just depend on it to be initialized from PT
    relative_position_bias_index =
      Axon.param("relative_position_index", {h * w + 1, h * w + 1}, initializer: :zeros)

    Axon.layer(
      fn table, idx, _opts ->
        index = Nx.reshape(idx, {:auto})
        shape = {h * w + 1, h * w + 1, :auto}

        table
        |> Nx.take(Nx.as_type(index, {:s, 64}))
        |> Nx.reshape(shape)
        |> Nx.transpose(axes: [2, 0, 1])
      end,
      [relative_position_bias_table, relative_position_bias_index],
      name: name
    )
  end

  defp self_output(hidden_states, config, opts) do
    name = opts[:name]

    hidden_states
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
  end

  defp intermediate(hidden_states, config, opts) do
    name = opts[:name]

    hidden_states
    |> Axon.dense(config.intermediate_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.activation(config.hidden_act, name: join(name, "dropout"))
  end

  defp output(hidden_states, config, opts) do
    name = opts[:name]

    hidden_states
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
  end

  defp pooler(hidden_states, config, opts) do
    name = opts[:name]

    if config.use_mean_pooling do
      # We use a global average pooling layer with layer
      # normalization of the patch_tokens
      hidden_states
      |> get_patch_tokens()
      |> Axon.global_avg_pool(channels: :last, name: join(name, "global_avg_pool"))
      |> Axon.layer_norm(epsilon: config.layer_norm_eps, name: join(name, "layernorm"))
    else
      # Otherwise we get the final hidden state of the [CLS]
      # token
      get_cls_hidden_state(hidden_states)
    end
  end

  defp get_patch_tokens(%Axon{} = states) do
    Axon.nx(states, fn x -> x[[0..-1//1, 1..-1//1, 0..-1//1]] end)
  end

  defp get_cls_hidden_state(%Axon{} = states) do
    Axon.nx(states, fn x -> x[[0..-1//1, 0]] end)
  end

  defp linspace(start, finish, steps) do
    step_size = (finish - start) / (steps - 1)
    outs = Nx.iota({steps})
    Nx.multiply(step_size, outs)
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.initializer_range)
  end

  defp shape(%Axon{output_shape: shape}), do: shape

  defp join(lhs, rhs), do: lhs <> "." <> rhs

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.atomize_values(["hidden_act"])
      |> Shared.cast_common_values()
      |> Shared.data_into_config(config)
    end
  end
end
