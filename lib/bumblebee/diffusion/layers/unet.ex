defmodule Bumblebee.Diffusion.Layers.UNet do
  @moduledoc false

  alias Bumblebee.Layers
  alias Bumblebee.Diffusion

  import Bumblebee.Utils.Model, only: [join: 2]

  @doc """
  Adds a feed-forward network on top of U-Net timestep embeddings.

  ## Options

    * `:name` - the base layer name

    * `:activation` - the activation used in-between the dense layers.
      Defaults to `:silu`

  """
  def timestep_embedding_mlp(sample, embedding_size, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, activation: :silu])
    name = opts[:name]

    sample
    |> Axon.dense(embedding_size, name: join(name, "linear_1"))
    |> Axon.activation(opts[:activation])
    |> Axon.dense(embedding_size, name: join(name, "linear_2"))
  end

  @doc """
  Adds a U-Net downsample block of the given type to the network.
  """
  def down_block_2d(
        :cross_attention_down_block,
        sample,
        timestep_embeds,
        encoder_hidden_state,
        opts
      ) do
    down_block_2d(sample, timestep_embeds, encoder_hidden_state, opts)
  end

  def down_block_2d(:down_block, sample, timestep_embeds, _encoder_hidden_state, opts) do
    down_block_2d(sample, timestep_embeds, nil, opts)
  end

  @doc """
  Adds a U-Net upsample block of the given type to the network.
  """
  def up_block_2d(
        :cross_attention_up_block,
        sample,
        timestep_embeds,
        residuals,
        encoder_hidden_state,
        opts
      ) do
    up_block_2d(sample, timestep_embeds, residuals, encoder_hidden_state, opts)
  end

  def up_block_2d(:up_block, sample, timestep_embeds, residuals, _encoder_hidden_state, opts) do
    up_block_2d(sample, timestep_embeds, residuals, nil, opts)
  end

  @doc """
  Adds U-Net downsample block to the network.

  When `encoder_hidden_state` is not `nil`, applies cross-attention.
  """
  def down_block_2d(hidden_state, timestep_embeds, encoder_hidden_state, opts \\ []) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_epsilon = opts[:resnet_epsilon] || 1.0e-6
    resnet_activation = opts[:resnet_activation] || :swish
    resnet_num_groups = opts[:resnet_num_groups] || 32
    num_attention_heads = opts[:num_attention_heads] || 1
    output_scale_factor = opts[:output_scale_factor] || 1.0
    downsample_padding = opts[:downsample_padding] || [{1, 1}, {1, 1}]
    add_downsample = Keyword.get(opts, :add_downsample, true)
    name = opts[:name]

    state = {hidden_state, {}}

    {hidden_state, output_states} =
      for idx <- 0..(num_layers - 1), reduce: state do
        {hidden_state, output_states} ->
          in_channels = if(idx == 0, do: in_channels, else: out_channels)

          hidden_state =
            Diffusion.Layers.resnet_block(
              hidden_state,
              in_channels,
              out_channels,
              timestep_embeds: timestep_embeds,
              epsilon: resnet_epsilon,
              num_groups: resnet_num_groups,
              dropout: dropout,
              activation: resnet_activation,
              output_scale_factor: output_scale_factor,
              name: join(name, "resnets.#{idx}")
            )

          hidden_state =
            if encoder_hidden_state do
              spatial_transformer(hidden_state, encoder_hidden_state,
                hidden_size: out_channels,
                num_heads: num_attention_heads,
                depth: 1,
                name: join(name, "attentions.#{idx}")
              )
            else
              hidden_state
            end

          {hidden_state, Tuple.append(output_states, {hidden_state, out_channels})}
      end

    if add_downsample do
      hidden_state =
        Diffusion.Layers.downsample_2d(hidden_state, out_channels,
          padding: downsample_padding,
          name: join(name, "downsamplers.0")
        )

      {hidden_state, Tuple.append(output_states, {hidden_state, out_channels})}
    else
      {hidden_state, output_states}
    end
  end

  @doc """
  Adds U-Net upsample block to the network.

  When `encoder_hidden_state` is not `nil`, applies cross-attention.
  """
  def up_block_2d(
        hidden_state,
        timestep_embeds,
        residuals,
        encoder_hidden_state,
        opts
      ) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_epsilon = opts[:resnet_epsilon] || 1.0e-6
    resnet_activation = opts[:resnet_activation] || :swish
    resnet_num_groups = opts[:resnet_num_groups] || 32
    num_attention_heads = opts[:num_attention_heads] || 1
    output_scale_factor = opts[:output_scale_factor] || 1.0
    add_upsample = Keyword.get(opts, :add_upsample, true)
    name = opts[:name]

    ^num_layers = length(residuals)

    hidden_state =
      for {{residual, residual_channels}, idx} <- Enum.with_index(residuals),
          reduce: hidden_state do
        hidden_state ->
          in_channels = if(idx == 0, do: in_channels, else: out_channels)

          hidden_state =
            Axon.concatenate([hidden_state, residual], axis: 1)
            |> Diffusion.Layers.resnet_block(
              in_channels + residual_channels,
              out_channels,
              timestep_embeds: timestep_embeds,
              epsilon: resnet_epsilon,
              num_groups: resnet_num_groups,
              dropout: dropout,
              activation: resnet_activation,
              output_scale_factor: output_scale_factor,
              name: join(name, "resnets.#{idx}")
            )

          if encoder_hidden_state do
            spatial_transformer(hidden_state, encoder_hidden_state,
              hidden_size: out_channels,
              num_heads: num_attention_heads,
              depth: 1,
              name: join(name, "attentions.#{idx}")
            )
          else
            hidden_state
          end
      end

    if add_upsample do
      Diffusion.Layers.upsample_2d(hidden_state, out_channels, name: join(name, "upsamplers.0"))
    else
      hidden_state
    end
  end

  @doc """
  Adds a U-Net middle block with cross attention to the network.
  """
  def mid_cross_attention_block_2d(
        hidden_state,
        timestep_embeds,
        encoder_hidden_state,
        opts \\ []
      ) do
    channels = opts[:channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_epsilon = opts[:resnet_epsilon] || 1.0e-6
    resnet_activation = opts[:resnet_activation] || :swish
    resnet_num_groups = opts[:resnet_num_groups] || 32
    num_attention_heads = opts[:num_attention_heads] || 1
    output_scale_factor = opts[:output_scale_factor] || 1.0
    name = opts[:name]

    resnet_block_opts = [
      epsilon: resnet_epsilon,
      num_groups: resnet_num_groups,
      dropout: dropout,
      activation: resnet_activation,
      output_scale_factor: output_scale_factor
    ]

    hidden_state =
      Diffusion.Layers.resnet_block(
        hidden_state,
        channels,
        channels,
        resnet_block_opts ++ [timestep_embeds: timestep_embeds, name: join(name, "resnets.0")]
      )

    for idx <- 0..(num_layers - 1), reduce: hidden_state do
      hidden_state ->
        hidden_state
        |> spatial_transformer(
          encoder_hidden_state,
          hidden_size: channels,
          num_heads: num_attention_heads,
          depth: 1,
          name: join(name, "attentions.#{idx}")
        )
        |> Diffusion.Layers.resnet_block(
          channels,
          channels,
          resnet_block_opts ++
            [timestep_embeds: timestep_embeds, name: join(name, "resnets.#{idx + 1}")]
        )
    end
  end

  defp spatial_transformer(hidden_state, cross_hidden_state, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    depth = opts[:depth] || 1
    dropout = opts[:dropout] || 0.0
    name = opts[:name]

    residual = hidden_state

    hidden_state
    |> Axon.group_norm(32, epsilon: 1.0e-6, name: join(name, "norm"))
    |> Axon.conv(hidden_size,
      kernel_size: 1,
      strides: 1,
      padding: :valid,
      name: join(name, "proj_in")
    )
    |> then(
      &Axon.layer(
        fn x, x_in, _opts ->
          {b, c, h, w} = Nx.shape(x_in)
          x |> Nx.transpose(axes: [0, 2, 3, 1]) |> Nx.reshape({b, h * w, c})
        end,
        [&1, residual]
      )
    )
    |> spatial_transformer_blocks(cross_hidden_state,
      hidden_size: hidden_size,
      num_heads: num_heads,
      dropout: dropout,
      depth: depth,
      name: name
    )
    |> then(
      &Axon.layer(
        fn x, x_in, _opts ->
          {b, c, h, w} = Nx.shape(x_in)
          x |> Nx.reshape({b, h, w, c}) |> Nx.transpose(axes: [0, 3, 1, 2])
        end,
        [&1, residual]
      )
    )
    |> Axon.conv(hidden_size,
      kernel_size: 1,
      strides: 1,
      padding: :valid,
      name: join(name, "proj_out")
    )
    |> Axon.add(residual)
  end

  defp spatial_transformer_blocks(hidden_state, cross_hidden_state, opts) do
    name = opts[:name]

    for idx <- 0..(opts[:depth] - 1), reduce: hidden_state do
      hidden_state ->
        transformer_block(hidden_state, cross_hidden_state,
          hidden_size: opts[:hidden_size],
          num_heads: opts[:num_heads],
          dropout: opts[:dropout],
          name: join(name, "transformer_blocks.#{idx}")
        )
    end
  end

  defp transformer_block(hidden_state, cross_hidden_state, opts) do
    name = opts[:name]

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(name: join(name, "norm1"), channel_index: 2)
      |> attention(nil,
        hidden_size: opts[:hidden_size],
        num_heads: opts[:num_heads],
        dropout: opts[:dropout],
        name: join(name, "attn1")
      )
      |> Axon.add(residual)

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(name: join(name, "norm2"), channel_index: 2)
      |> attention(cross_hidden_state,
        hidden_size: opts[:hidden_size],
        num_heads: opts[:num_heads],
        dropout: opts[:dropout],
        name: join(name, "attn2")
      )
      |> Axon.add(residual)

    residual = hidden_state

    hidden_state
    |> Axon.layer_norm(name: join(name, "norm3"), channel_index: 2)
    |> feed_forward_geglu(opts[:hidden_size], dropout: opts[:dropout], name: join(name, "ff"))
    |> Axon.add(residual)
  end

  defp attention(hidden_state, cross_hidden_state, opts) do
    name = opts[:name]
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads] || 8
    dropout = opts[:dropout] || 0.0

    cross_hidden_state = cross_hidden_state || hidden_state

    query =
      hidden_state
      |> Axon.dense(hidden_size, use_bias: false, name: join(name, "to_q"))
      |> Layers.split_heads(num_heads)

    key =
      cross_hidden_state
      |> Axon.dense(hidden_size, use_bias: false, name: join(name, "to_k"))
      |> Layers.split_heads(num_heads)

    value =
      cross_hidden_state
      |> Axon.dense(hidden_size, use_bias: false, name: join(name, "to_v"))
      |> Layers.split_heads(num_heads)

    attention_weights =
      Layers.attention_weights(query, key, Axon.constant(Nx.tensor(0)))
      |> Axon.dropout(rate: dropout, name: join(name, "dropout"))

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()

    attention_output
    |> Axon.dense(hidden_size, name: join(name, "to_out.0"))
    |> Axon.dropout(rate: dropout)
  end

  # A feed-forward network with GEGLU nonlinearity as in https://arxiv.org/abs/2002.05202
  defp feed_forward_geglu(x, size, opts) do
    dropout = opts[:dropout] || 0.0
    name = opts[:name]

    inner_size = 4 * size

    x
    |> geglu(inner_size, name: join(name, "net.0"))
    |> Axon.dropout(rate: dropout, name: join(name, "net.1"))
    |> Axon.dense(size, name: join(name, "net.2"))
  end

  defp geglu(x, size, opts) do
    name = opts[:name]

    {x, gate} =
      x
      |> Axon.dense(size * 2, name: join(name, "proj"))
      |> Axon.split(2, axis: -1)

    Axon.multiply(x, Axon.gelu(gate))
  end
end
