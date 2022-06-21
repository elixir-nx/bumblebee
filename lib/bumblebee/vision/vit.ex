defmodule Bumblebee.Vision.Vit do
  @common_keys [:id2label, :label2id, :num_labels, :output_hidden_states]

  @moduledoc """
  Models based on the ViT architecture.

  ## Architectures

    * `:base` - plain ViT without any head on top

    * `:for_image_classification` - ViT with a classification head.
      The head consists of a single dense layer on top of the pooled
      features

    * `:for_masked_image_modeling` - BEiT with a language modeling
      head on top for predicting visual tokens

  ## Config

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

    * `:qkv_bias` - whether to use bias in query, key, and value
      projections

    * `:encoder_stride` - factor to increase the spatial resolution by
    in the decoder head for masked image modeling

  ### Common Options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}

  ## References

    * [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
  """

  alias Bumblebee.Layers
  alias Bumblebee.Shared

  defstruct [
              architecture: :base,
              hidden_size: 768,
              num_hidden_layers: 12,
              num_attention_heads: 12,
              intermediate_size: 3072,
              hidden_act: :gelu,
              hidden_dropout_prob: 0.0,
              attention_probs_dropout_prob: 0.0,
              initializer_range: 0.02,
              layer_norm_eps: 1.0e-12,
              is_encoder_decoder: false,
              image_size: 224,
              patch_size: 16,
              num_channels: 3,
              qkv_bias: true,
              encoder_stride: 16
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

  @impl true
  def architectures(), do: [:base, :for_image_classification, :for_masked_image_modeling]

  @impl true
  def base_model_prefix(), do: "vit"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def model(%__MODULE__{architecture: :for_image_classification} = config) do
    outputs = vit(config, name: "vit")

    logits =
      outputs.last_hidden_state
      |> Layers.take_head_layer(axis: 1, name: join("vit", "head"))
      |> Axon.dense(config.num_labels,
        kernel_initializer: kernel_initializer(config),
        name: "classifier"
      )

    Axon.container(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  # TODO: This requires a PixelShuffle implementation in Axon
  # def model(%__MODULE__{architecture: :for_masked_image_modeling} = config) do    
  # end

  def model(%__MODULE__{architecture: :base} = config) do
    config
    |> vit()
    |> Axon.container()
  end

  defp vit(config, opts \\ []) do
    name = opts[:name]

    input_shape = {nil, config.num_channels, config.image_size, config.image_size}
    pixel_values = Axon.input(input_shape, "pixel_values")

    hidden_states = embeddings(pixel_values, config, name: join(name, "embeddings"))

    {hidden_states, all_hidden_states, all_attentions} =
      encoder(hidden_states, config, name: join(name, "encoder"))

    last_hidden_state =
      hidden_states
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "layernorm")
      )

    pooled = pooler(last_hidden_state, config, name: join(name, "pooler"))

    %{
      last_hidden_state: last_hidden_state,
      pooler_output: pooled,
      hidden_states: all_hidden_states,
      attentions: all_attentions
    }
  end

  defp embeddings(pixel_values, config, opts) do
    name = opts[:name]

    pixel_values
    |> patch_embeddings(config, name: join(name, "patch_embeddings"))
    |> position_embeddings(config, name: name)
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
  end

  defp patch_embeddings(pixel_values, config, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(config.hidden_size,
      kernel_size: config.patch_size,
      strides: config.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "projection")
    )
    |> Axon.nx(&Nx.transpose(&1, axes: [0, 2, 3, 1]))
    |> Axon.reshape({:auto, config.hidden_size}, name: join(name, "reshape"))
  end

  defp position_embeddings(embeds, config, opts) do
    name = opts[:name]

    num_patches =
      div(config.image_size, config.patch_size) * div(config.image_size, config.patch_size)

    cls_token = Axon.param("cls_token", {1, 1, config.hidden_size}, initializer: :zeros)

    position_embeddings =
      Axon.param("position_embeddings", {1, num_patches + 1, config.hidden_size},
        initializer: :zeros
      )

    Axon.layer(
      fn inp, tok, pos, _opts ->
        batch_size = Nx.axis_size(inp, 0)
        tok = Nx.broadcast(tok, {batch_size, 1, config.hidden_size})

        Nx.concatenate([tok, inp], axis: 1)
        |> Nx.add(pos)
      end,
      [embeds, cls_token, position_embeddings],
      name: name
    )
  end

  defp encoder(hidden_states, config, opts) do
    name = opts[:name]

    encoder_layer_collection(hidden_states, config, name: name)
  end

  defp encoder_layer_collection(hidden_states, config, opts) do
    name = opts[:name]

    last_hidden_state = hidden_states
    all_hidden_states = {last_hidden_state}
    all_attentions = {}

    for idx <- 0..(config.num_hidden_layers - 1),
        reduce: {last_hidden_state, all_hidden_states, all_attentions} do
      {lhs, states, attns} ->
        layer_name = join(name, "layer.#{idx}")
        {next_state, next_attention} = encoder_layer(lhs, config, name: layer_name)
        {next_state, Tuple.append(states, next_state), Tuple.append(attns, next_attention)}
    end
  end

  defp encoder_layer(hidden_states, config, opts) do
    name = opts[:name]

    {attention_output, attention_weights} =
      hidden_states
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "layernorm_before")
      )
      |> attention(config, name: join(name, "attention"))

    attention_output = Axon.add(attention_output, hidden_states)

    layer_output =
      attention_output
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "layernorm_after")
      )
      |> intermediate(config, name: join(name, "intermediate"))
      |> output(attention_output, config, name: join(name, "output"))

    {layer_output, attention_weights}
  end

  defp attention(hidden_states, config, opts) do
    name = opts[:name]

    {attention_output, attention_weights} =
      self_attention(hidden_states, config, name: join(name, "attention"))

    attention_output =
      self_output(attention_output, hidden_states, config, name: join(name, "output"))

    {attention_output, attention_weights}
  end

  defp self_attention(hidden_states, config, opts) do
    name = opts[:name]

    head_dim = div(config.hidden_size, config.num_attention_heads)

    query_states =
      hidden_states
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        use_bias: config.qkv_bias,
        name: join(name, "query")
      )
      |> Axon.reshape({:auto, config.num_attention_heads, head_dim})

    key_states =
      hidden_states
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        use_bias: config.qkv_bias,
        name: join(name, "key")
      )
      |> Axon.reshape({:auto, config.num_attention_heads, head_dim})

    value_states =
      hidden_states
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        use_bias: config.qkv_bias,
        name: join(name, "value")
      )
      |> Axon.reshape({:auto, config.num_attention_heads, head_dim})

    attention_bias = Axon.constant(Nx.tensor(0))

    attention_weights =
      Axon.layer(&Layers.attention_weights/4, [query_states, key_states, attention_bias])

    attention_weights =
      Axon.dropout(attention_weights,
        rate: config.attention_probs_dropout_prob,
        name: join(name, "dropout")
      )

    attention_output = Axon.layer(&Layers.attention_output/3, [attention_weights, value_states])

    attention_output =
      Axon.reshape(attention_output, {:auto, config.num_attention_heads * head_dim})

    {attention_output, attention_weights}
  end

  defp self_output(hidden_states, _input_tensor, config, opts) do
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
    |> Axon.activation(config.hidden_act)
  end

  defp output(hidden_states, attention_output, config, opts) do
    name = opts[:name]

    hidden_states
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
    |> Axon.add(attention_output, name: join(name, "residual"))
  end

  defp pooler(hidden_states, config, opts) do
    name = opts[:name]

    hidden_states
    |> Layers.take_head_layer(axis: 1, name: join(name, "head"))
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.tanh(name: join(name, "tanh"))
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.initializer_range)
  end

  defp join(nil, rhs), do: rhs

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
