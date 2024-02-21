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
    |> Axon.dense(embedding_size, name: join(name, "intermediate"))
    |> Axon.activation(opts[:activation])
    |> Axon.dense(embedding_size, name: join(name, "output"))
  end

  @doc """
  Adds a U-Net downsample block of the given type to the network.
  """
  def down_block_2d(
        :cross_attention_down_block,
        sample,
        timestep_embedding,
        encoder_hidden_state,
        opts
      ) do
    down_block_2d(sample, timestep_embedding, encoder_hidden_state, opts)
  end

  def down_block_2d(:down_block, sample, timestep_embedding, _encoder_hidden_state, opts) do
    down_block_2d(sample, timestep_embedding, nil, opts)
  end

  @doc """
  Adds a U-Net upsample block of the given type to the network.
  """
  def up_block_2d(
        :cross_attention_up_block,
        sample,
        timestep_embedding,
        residuals,
        encoder_hidden_state,
        opts
      ) do
    up_block_2d(sample, timestep_embedding, residuals, encoder_hidden_state, opts)
  end

  def up_block_2d(
        :up_block,
        sample,
        timestep_embedding,
        residuals,
        _encoder_hidden_state,
        opts
      ) do
    up_block_2d(sample, timestep_embedding, residuals, nil, opts)
  end

  @doc """
  Adds U-Net downsample block to the network.

  When `encoder_hidden_state` is not `nil`, applies cross-attention.
  """
  def down_block_2d(hidden_state, timestep_embedding, encoder_hidden_state, opts \\ []) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    dropout = opts[:dropout] || 0.0
    depth = opts[:depth] || 1
    activation = opts[:activation] || :swish
    norm_epsilon = opts[:norm_epsilon] || 1.0e-6
    norm_num_groups = opts[:norm_num_groups] || 32
    num_attention_heads = opts[:num_attention_heads] || 1
    use_linear_projection = Keyword.get(opts, :use_linear_projection, false)
    output_scale_factor = opts[:output_scale_factor] || 1.0
    downsample_padding = opts[:downsample_padding] || [{1, 1}, {1, 1}]
    add_downsample = Keyword.get(opts, :add_downsample, true)
    name = opts[:name]

    state = {hidden_state, {}}

    {hidden_state, output_states} =
      for idx <- 0..(depth - 1), reduce: state do
        {hidden_state, output_states} ->
          in_channels = if(idx == 0, do: in_channels, else: out_channels)

          hidden_state =
            Diffusion.Layers.residual_block(
              hidden_state,
              in_channels,
              out_channels,
              timestep_embedding: timestep_embedding,
              norm_epsilon: norm_epsilon,
              norm_num_groups: norm_num_groups,
              dropout: dropout,
              activation: activation,
              output_scale_factor: output_scale_factor,
              name: name |> join("residual_blocks") |> join(idx)
            )

          hidden_state =
            if encoder_hidden_state do
              transformer_2d(hidden_state, encoder_hidden_state,
                hidden_size: out_channels,
                num_heads: num_attention_heads,
                use_linear_projection: use_linear_projection,
                depth: 1,
                name: name |> join("transformers") |> join(idx)
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
          name: join(name, "downsamples.0")
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
        timestep_embedding,
        residuals,
        encoder_hidden_state,
        opts
      ) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    dropout = opts[:dropout] || 0.0
    depth = opts[:depth] || 1
    activation = opts[:activation] || :swish
    norm_epsilon = opts[:norm_epsilon] || 1.0e-6
    norm_num_groups = opts[:norm_num_groups] || 32
    num_attention_heads = opts[:num_attention_heads] || 1
    use_linear_projection = Keyword.get(opts, :use_linear_projection, false)
    output_scale_factor = opts[:output_scale_factor] || 1.0
    add_upsample = Keyword.get(opts, :add_upsample, true)
    name = opts[:name]

    ^depth = length(residuals)

    hidden_state =
      for {{residual, residual_channels}, idx} <- Enum.with_index(residuals),
          reduce: hidden_state do
        hidden_state ->
          in_channels = if(idx == 0, do: in_channels, else: out_channels)

          hidden_state =
            Axon.concatenate([hidden_state, residual], axis: -1)
            |> Diffusion.Layers.residual_block(
              in_channels + residual_channels,
              out_channels,
              timestep_embedding: timestep_embedding,
              norm_epsilon: norm_epsilon,
              norm_num_groups: norm_num_groups,
              dropout: dropout,
              activation: activation,
              output_scale_factor: output_scale_factor,
              name: name |> join("residual_blocks") |> join(idx)
            )

          if encoder_hidden_state do
            transformer_2d(hidden_state, encoder_hidden_state,
              hidden_size: out_channels,
              num_heads: num_attention_heads,
              use_linear_projection: use_linear_projection,
              depth: 1,
              name: name |> join("transformers") |> join(idx)
            )
          else
            hidden_state
          end
      end

    if add_upsample do
      Diffusion.Layers.upsample_2d(hidden_state, out_channels, name: join(name, "upsamples.0"))
    else
      hidden_state
    end
  end

  @doc """
  Adds a U-Net middle block with cross attention to the network.
  """
  def mid_cross_attention_block_2d(
        hidden_state,
        timestep_embedding,
        encoder_hidden_state,
        opts \\ []
      ) do
    channels = opts[:channels]
    dropout = opts[:dropout] || 0.0
    depth = opts[:depth] || 1
    norm_epsilon = opts[:norm_epsilon] || 1.0e-6
    activation = opts[:activation] || :swish
    norm_num_groups = opts[:norm_num_groups] || 32
    num_attention_heads = opts[:num_attention_heads] || 1
    use_linear_projection = Keyword.get(opts, :use_linear_projection, false)
    output_scale_factor = opts[:output_scale_factor] || 1.0
    name = opts[:name]

    residual_block_opts = [
      epsilon: norm_epsilon,
      num_groups: norm_num_groups,
      dropout: dropout,
      activation: activation,
      output_scale_factor: output_scale_factor
    ]

    hidden_state =
      Diffusion.Layers.residual_block(
        hidden_state,
        channels,
        channels,
        residual_block_opts ++
          [timestep_embedding: timestep_embedding, name: join(name, "residual_blocks.0")]
      )

    for idx <- 0..(depth - 1), reduce: hidden_state do
      hidden_state ->
        hidden_state
        |> transformer_2d(
          encoder_hidden_state,
          hidden_size: channels,
          num_heads: num_attention_heads,
          use_linear_projection: use_linear_projection,
          depth: 1,
          name: name |> join("transformers") |> join(idx)
        )
        |> Diffusion.Layers.residual_block(
          channels,
          channels,
          residual_block_opts ++
            [
              timestep_embedding: timestep_embedding,
              name: name |> join("residual_blocks") |> join(idx + 1)
            ]
        )
    end
  end

  defp transformer_2d(hidden_state, cross_hidden_state, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    use_linear_projection = opts[:use_linear_projection]
    depth = opts[:depth] || 1
    dropout = opts[:dropout] || 0.0
    name = opts[:name]

    shortcut = hidden_state

    flatten_spatial =
      &Axon.layer(
        fn hidden_state, shortcut, _opts ->
          {b, h, w, c} = Nx.shape(shortcut)
          Nx.reshape(hidden_state, {b, h * w, c})
        end,
        [&1, shortcut]
      )

    unflatten_spatial =
      &Axon.layer(
        fn hidden_state, shortcut, _opts ->
          Nx.reshape(hidden_state, Nx.shape(shortcut))
        end,
        [&1, shortcut]
      )

    hidden_state
    |> Axon.group_norm(32, epsilon: 1.0e-6, name: join(name, "norm"))
    |> then(fn hidden_state ->
      if use_linear_projection do
        hidden_state
        |> flatten_spatial.()
        |> Axon.dense(hidden_size, name: join(name, "input_projection"))
      else
        hidden_state
        |> Axon.conv(hidden_size,
          kernel_size: 1,
          strides: 1,
          padding: :valid,
          name: join(name, "input_projection")
        )
        |> flatten_spatial.()
      end
    end)
    |> Layers.Transformer.blocks(
      cross_hidden_state: cross_hidden_state,
      num_blocks: depth,
      num_attention_heads: num_heads,
      hidden_size: hidden_size,
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      layer_norm: [
        epsilon: 1.0e-5
      ],
      dropout_rate: dropout,
      ffn: &ffn_geglu(&1, 4 * hidden_size, hidden_size, dropout: dropout, name: &2),
      block_type: :norm_first,
      name: join(name, "blocks")
    )
    |> then(fn %{hidden_state: hidden_state} ->
      if use_linear_projection do
        hidden_state
        |> Axon.dense(hidden_size, name: join(name, "output_projection"))
        |> unflatten_spatial.()
      else
        hidden_state
        |> unflatten_spatial.()
        |> Axon.conv(hidden_size,
          kernel_size: 1,
          strides: 1,
          padding: :valid,
          name: join(name, "output_projection")
        )
      end
    end)
    |> Axon.add(shortcut)
  end

  # A feed-forward network with GEGLU nonlinearity as in https://arxiv.org/abs/2002.05202
  defp ffn_geglu(x, intermediate_size, output_size, opts) do
    name = opts[:name]
    dropout = opts[:dropout] || 0.0

    {x, gate} =
      x
      |> Axon.dense(intermediate_size * 2, name: join(name, "intermediate"))
      |> Axon.split(2, axis: -1)

    x = Axon.multiply(x, Axon.gelu(gate))

    x
    |> Axon.dropout(rate: dropout, name: join(name, "dropout"))
    |> Axon.dense(output_size, name: join(name, "output"))
  end
end
