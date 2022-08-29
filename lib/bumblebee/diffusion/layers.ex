defmodule Bumblebee.Diffusion.Layers do
  alias Bumblebee.Layers

  import Bumblebee.Utils.Model, only: [join: 2]
  import Nx.Defn

  ## Embeddings

  def timesteps(timesteps, opts \\ []) do
    Axon.layer(&timesteps_impl/2, [timesteps], opts)
  end

  defnp timesteps_impl(timesteps, opts \\ []) do
    opts =
      keyword!(opts, [
        :num_channels,
        flip_sin_to_cos: false,
        downscale_freq_shift: 1,
        scale: 1,
        max_period: 10_000,
        mode: :train
      ])

    embedding_dim = opts[:num_channels]
    half_dim = div(embedding_dim, 2)
    exponent = get_exponent(opts[:max_period], half_dim, opts[:downscale_freq_shift])

    emb = Nx.exp(exponent)
    emb = Nx.new_axis(timesteps, -1) * Nx.new_axis(emb, 0)

    emb = opts[:scale] * emb
    emb = Nx.concatenate([Nx.sin(emb), Nx.cos(emb)], axis: -1)

    emb =
      if opts[:flip_sin_to_cos] do
        Nx.concatenate([emb[[0..-1//1, half_dim..-1//1]], emb[[0..-1//1, 0..(half_dim - 1)]]],
          axis: -1
        )
      else
        emb
      end

    # TODO: Pad
    emb
  end

  deftransformp get_exponent(max_period, half_dim, downscale_freq_shift) do
    max_period
    |> Nx.log()
    |> Nx.negate()
    |> Nx.multiply(Nx.iota({half_dim}))
    |> Nx.divide(Nx.subtract(half_dim, downscale_freq_shift))
  end

  def timestep_embedding(sample, time_embed_dim, opts \\ []) do
    name = opts[:name]
    activation = opts[:act_fn] || :silu

    sample
    |> Axon.dense(time_embed_dim, name: join(name, "linear_1"))
    |> Axon.activation(activation)
    |> Axon.dense(time_embed_dim, name: join(name, "linear_2"))
  end

  @doc """
  Maps the given block type to the corresponding
  UNet block.
  """
  def apply_unet_block("DownEncoderBlock2D", sample, block_opts) do
    apply(__MODULE__, :down_encoder_block_2d, [sample, block_opts])
  end

  def apply_unet_block("UpDecoderBlock2D", sample, block_opts) do
    apply(__MODULE__, :up_decoder_block_2d, [sample, block_opts])
  end

  def apply_unet_block("CrossAttnDownBlock2D", inputs) do
    apply(__MODULE__, :cross_attn_down_block_2d, inputs)
  end

  def apply_unet_block("DownBlock2D", [sample, timestep_embedding, _, block_opts]) do
    apply(__MODULE__, :down_block_2d, [sample, timestep_embedding, block_opts])
  end

  def apply_unet_block("CrossAttnUpBlock2D", inputs) do
    apply(__MODULE__, :cross_attn_up_block_2d, inputs)
  end

  def apply_unet_block("UpBlock2D", [sample, timestep_embedding, res_samples, _, block_opts]) do
    apply(__MODULE__, :up_block_2d, [sample, timestep_embedding, res_samples, block_opts])
  end

  ## UNet Blocks

  @doc """
  Adds a UNet down-encoder block to the network.
  """
  def down_encoder_block_2d(hidden_state, opts \\ []) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    downsample_padding = opts[:downsample_padding] || [{1, 1}, {1, 1}]
    output_scale_factor = opts[:output_scale_factor] || 1.0
    add_downsample = Keyword.get(opts, :add_downsample, true)
    name = opts[:name]

    acc = {hidden_state, in_channels}

    {hidden_state, _} =
      for idx <- 0..(num_layers - 1), reduce: acc do
        {hidden_state, in_channels} ->
          block_opts = [
            in_channels: in_channels,
            out_channels: out_channels,
            eps: resnet_eps,
            groups: resnet_groups,
            dropout: dropout,
            non_linearity: resnet_act_fn,
            output_scale_factor: output_scale_factor,
            name: join(name, "resnets.#{idx}")
          ]

          {resnet_block(hidden_state, nil, block_opts), out_channels}
      end

    if add_downsample do
      downsample_2d(hidden_state,
        use_conv: true,
        out_channels: out_channels,
        padding: downsample_padding,
        name: join(name, "downsamplers.0")
      )
    else
      hidden_state
    end
  end

  @doc """
  Applies a UNet up-decoder 2D block to the network.
  """
  def up_decoder_block_2d(hidden_state, opts \\ []) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    output_scale_factor = opts[:output_scale_factor] || 1.0
    add_upsample = Keyword.get(opts, :add_upsample, true)
    name = opts[:name]

    acc = {hidden_state, in_channels}

    {hidden_state, _} =
      for idx <- 0..(num_layers - 1), reduce: acc do
        {hidden_state, in_channels} ->
          block_opts = [
            in_channels: in_channels,
            out_channels: out_channels,
            eps: resnet_eps,
            groups: resnet_groups,
            non_linearity: resnet_act_fn,
            output_scale_factor: output_scale_factor,
            name: join(name, "resnets.#{idx}")
          ]

          {resnet_block(hidden_state, nil, block_opts), out_channels}
      end

    if add_upsample do
      upsample_2d(hidden_state,
        use_conv: true,
        out_channels: out_channels,
        name: join(name, "upsamplers.0")
      )
    else
      hidden_state
    end
  end

  @doc """
  Applies a UNet mid-block to the network.
  """
  def mid_block_2d(hidden_state, opts \\ []) do
    in_channels = opts[:in_channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    output_scale_factor = opts[:output_scale_factor] || 1.0
    attn_num_head_channels = opts[:attn_num_head_channels] || 1
    name = opts[:name]

    block_opts = [
      in_channels: in_channels,
      out_channels: in_channels,
      eps: resnet_eps,
      groups: resnet_groups,
      dropout: dropout,
      non_linearity: resnet_act_fn,
      output_scale_factor: output_scale_factor
    ]

    hidden_state = resnet_block(hidden_state, nil, block_opts ++ [name: join(name, "resnets.0")])

    for idx <- 0..(num_layers - 1), reduce: hidden_state do
      hidden_state ->
        hidden_state
        |> visual_attention_block(
          channels: in_channels,
          num_head_channels: attn_num_head_channels,
          eps: resnet_eps,
          num_groups: resnet_groups,
          name: join(name, "attentions.#{idx}")
        )
        |> resnet_block(nil, block_opts ++ [name: join(name, "resnets.#{idx + 1}")])
    end
  end

  def cross_attn_down_block_2d(
        hidden_state,
        timestep_embedding,
        encoder_hidden_states,
        opts \\ []
      ) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    temb_channels = opts[:temb_channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    attn_num_head_channels = opts[:attn_num_head_channels] || 1
    cross_attention_dim = opts[:cross_attention_dim] || 1280
    output_scale_factor = opts[:output_scale_factor] || 1.0
    downsample_padding = opts[:downsample_padding] || [{1, 1}, {1, 1}]
    add_downsample = Keyword.get(opts, :add_downsample, true)
    name = opts[:name]

    state = {hidden_state, {}, in_channels}

    {hidden_state, output_states, _} =
      for idx <- 0..(num_layers - 1), reduce: state do
        {hidden_state, output_states, in_channels} ->
          block_opts = [
            in_channels: in_channels,
            out_channels: out_channels,
            temb_channels: temb_channels,
            eps: resnet_eps,
            groups: resnet_groups,
            dropout: dropout,
            non_linearity: resnet_act_fn,
            output_scale_factor: output_scale_factor,
            name: join(name, "resnets.#{idx}")
          ]

          hidden_state =
            hidden_state
            |> resnet_block(timestep_embedding, block_opts)
            |> spatial_transformer(encoder_hidden_states,
              in_channels: out_channels,
              n_heads: attn_num_head_channels,
              d_head: div(out_channels, attn_num_head_channels),
              depth: 1,
              context_dim: cross_attention_dim,
              name: join(name, "attentions.#{idx}")
            )

          {hidden_state, Tuple.append(output_states, hidden_state), out_channels}
      end

    if add_downsample do
      hidden_state =
        downsample_2d(hidden_state,
          use_conv: true,
          out_channels: out_channels,
          padding: downsample_padding,
          name: join(name, "downsamplers.0")
        )

      {hidden_state, Tuple.append(output_states, hidden_state)}
    else
      {hidden_state, output_states}
    end
  end

  @doc """
  Adds a UNet down-block to the network.
  """
  def down_block_2d(hidden_state, timestep_embedding, opts \\ []) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    temb_channels = opts[:temb_channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    output_scale_factor = opts[:output_scale_factor] || 1.0
    downsample_padding = opts[:downsample_padding] || [{1, 1}, {1, 1}]
    add_downsample = Keyword.get(opts, :add_downsample, true)
    name = opts[:name]

    state = {hidden_state, {}, in_channels}

    {hidden_state, output_states, _} =
      for idx <- 0..(num_layers - 1), reduce: state do
        {hidden_state, output_states, in_channels} ->
          block_opts = [
            in_channels: in_channels,
            out_channels: out_channels,
            temb_channels: temb_channels,
            eps: resnet_eps,
            groups: resnet_groups,
            dropout: dropout,
            non_linearity: resnet_act_fn,
            output_scale_factor: output_scale_factor,
            name: join(name, "resnets.#{idx}")
          ]

          hidden_state = resnet_block(hidden_state, timestep_embedding, block_opts)
          {hidden_state, Tuple.append(output_states, hidden_state), out_channels}
      end

    if add_downsample do
      hidden_state =
        downsample_2d(hidden_state,
          use_conv: true,
          out_channels: out_channels,
          padding: downsample_padding,
          name: join(name, "downsamplers.0")
        )

      {hidden_state, Tuple.append(output_states, hidden_state)}
    else
      {hidden_state, output_states}
    end
  end

  @doc """
  Adds a mid-block with cross-attention to the network.
  """
  def mid_block_2d_cross_attn(hidden_state, timestep_embedding, encoder_hidden_states, opts \\ []) do
    in_channels = opts[:in_channels]
    temb_channels = opts[:temb_channels]
    dropout = opts[:dropout] || 0.0
    num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    attn_num_head_channels = opts[:attn_num_head_channels] || 1
    output_scale_factor = opts[:output_scale_factor] || 1.0
    cross_attention_dim = opts[:cross_attention_dim] || 1280
    name = opts[:name]

    block_opts = [
      in_channels: in_channels,
      out_channels: in_channels,
      temb_channels: temb_channels,
      eps: resnet_eps,
      groups: resnet_groups,
      dropout: dropout,
      non_linearity: resnet_act_fn,
      output_scale_factor: output_scale_factor
    ]

    hidden_state =
      resnet_block(
        hidden_state,
        timestep_embedding,
        block_opts ++ [name: join(name, "resnets.0")]
      )

    for idx <- 0..(num_layers - 1), reduce: hidden_state do
      hidden_state ->
        hidden_state
        |> spatial_transformer(
          encoder_hidden_states,
          in_channels: in_channels,
          n_heads: attn_num_head_channels,
          d_head: div(in_channels, attn_num_head_channels),
          depth: 1,
          context_dim: cross_attention_dim,
          name: join(name, "attentions.#{idx}")
        )
        |> resnet_block(
          timestep_embedding,
          block_opts ++ [name: join(name, "resnets.#{idx + 1}")]
        )
    end
  end

  @doc """
  Adds a UNet up-block with cross-attention to the network.
  """
  def cross_attn_up_block_2d(
        hidden_state,
        timestep_embedding,
        res_hidden_states,
        encoder_hidden_states,
        opts
      ) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    # TODO: Work this in
    # prev_output_channel = opts[:prev_output_channel]
    temb_channels = opts[:temb_channels]
    dropout = opts[:dropout] || 0.0
    # TODO: Work this in
    # num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    attn_num_head_channels = opts[:attn_num_head_channels] || 1
    cross_attention_dim = opts[:cross_attention_dim] || 1280
    output_scale_factor = opts[:output_scale_factor] || 1.0
    add_upsample = Keyword.get(opts, :add_upsample, true)
    name = opts[:name]

    {hidden_state, _} =
      for {res_hidden_state, idx} <- Enum.with_index(res_hidden_states),
          reduce: {hidden_state, in_channels} do
        {hidden_state, in_channels} ->
          block_opts = [
            in_channels: in_channels + in_channels,
            out_channels: out_channels,
            temb_channels: temb_channels,
            eps: resnet_eps,
            groups: resnet_groups,
            dropout: dropout,
            non_linearity: resnet_act_fn,
            output_scale_factor: output_scale_factor,
            name: join(name, "resnets.#{idx}")
          ]

          hidden_state =
            Axon.concatenate([hidden_state, res_hidden_state], axis: 1)
            |> resnet_block(timestep_embedding, block_opts)
            |> spatial_transformer(
              encoder_hidden_states,
              in_channels: out_channels,
              n_heads: attn_num_head_channels,
              d_head: div(out_channels, attn_num_head_channels),
              depth: 1,
              context_dim: cross_attention_dim,
              name: join(name, "attentions.#{idx}")
            )

          {hidden_state, out_channels}
      end

    if add_upsample do
      upsample_2d(hidden_state,
        use_conv: true,
        out_channels: out_channels,
        name: join(name, "upsamplers.0")
      )
    else
      hidden_state
    end
  end

  @doc """
  Adds a UNet up-block to the network.
  """
  def up_block_2d(hidden_state, timestep_embedding, res_hidden_states, opts \\ []) do
    in_channels = opts[:in_channels]
    # TODO: work this in
    # prev_output_channel = opts[:prev_output_channel]
    out_channels = opts[:out_channels]
    temb_channels = opts[:temb_channels]
    dropout = opts[:dropout] || 0.0
    # TODO: Work this in
    # num_layers = opts[:num_layers] || 1
    resnet_eps = opts[:resnet_eps] || 1.0e-6
    resnet_act_fn = opts[:resnet_act_fn] || :swish
    resnet_groups = opts[:resnet_groups] || 32
    output_scale_factor = opts[:output_scale_factor] || 1.0
    add_upsample = Keyword.get(opts, :add_upsample, true)
    name = opts[:name]

    {hidden_state, _} =
      for {res_hidden_state, idx} <- Enum.with_index(res_hidden_states),
          reduce: {hidden_state, in_channels} do
        {hidden_state, in_channels} ->
          block_opts = [
            in_channels: in_channels + in_channels,
            out_channels: out_channels,
            temb_channels: temb_channels,
            eps: resnet_eps,
            groups: resnet_groups,
            dropout: dropout,
            non_linearity: resnet_act_fn,
            output_scale_factor: output_scale_factor,
            name: join(name, "resnets.#{idx}")
          ]

          hidden_state =
            Axon.concatenate([hidden_state, res_hidden_state], axis: 1)
            |> resnet_block(timestep_embedding, block_opts)

          {hidden_state, out_channels}
      end

    if add_upsample do
      upsample_2d(hidden_state,
        use_conv: true,
        out_channels: out_channels,
        name: join(name, "upsamplers.0")
      )
    else
      hidden_state
    end
  end

  @doc """
  Adds an upsample layer to the network.
  """
  def upsample_2d(x, opts \\ []) do
    out_channels = opts[:out_channels]
    use_conv = opts[:use_conv]
    use_conv_transpose = opts[:use_conv_transpose]
    name = opts[:name]

    cond do
      use_conv_transpose ->
        Axon.conv_transpose(x, out_channels,
          kernel_size: 4,
          strides: 2,
          padding: [{1, 1}, {1, 1}],
          name: join(name, "conv")
        )

      use_conv ->
        x
        |> Axon.nx(fn x ->
          {_, _, h, w} = Nx.shape(x)
          Axon.Layers.resize(x, size: {2 * h, 2 * w}, mode: :nearest)
        end)
        |> Axon.conv(out_channels,
          kernel_size: 3,
          padding: [{1, 1}, {1, 1}],
          name: join(name, "conv")
        )

      true ->
        raise "uh oh"
    end
  end

  ## ResNet Blocks

  @doc """
  Adds a downsample block to the network.
  """
  def downsample_2d(x, opts \\ []) do
    out_channels = opts[:out_channels]
    use_conv = opts[:use_conv]
    padding = opts[:padding]
    name = opts[:name]
    stride = 2

    {x, padding} =
      cond do
        padding == 0 ->
          {Axon.nx(x, &Nx.pad(&1, 0.0, [{0, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 1, 0}])), :valid}

        padding == 1 ->
          {x, [{1, 1}, {1, 1}]}

        true ->
          {x, padding}
      end

    cond do
      use_conv ->
        Axon.conv(x, out_channels,
          kernel_size: 3,
          strides: stride,
          padding: padding,
          name: join(name, "conv")
        )

      true ->
        Axon.avg_pool(x, kernel_size: stride, strides: stride)
    end
  end

  @doc """
  Adds a ResNet block to the network.
  """
  def resnet_block(x, timestep_embedding, opts \\ []) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels] || in_channels
    temb_channels = opts[:temb_channels] || 512
    dropout = opts[:dropout] || 0.0
    groups = opts[:groups] || 32
    groups_out = opts[:groups_out] || groups
    eps = opts[:eps] || 1.0e-6
    non_linearity = opts[:non_linearity] || :swish
    output_scale_factor = opts[:output_scale_factor] || 1.0
    use_nin_shortcut = Keyword.get(opts, :use_nin_shortcut, in_channels != out_channels)
    name = opts[:name]

    h = x

    h =
      h
      |> Axon.group_norm(groups, epsilon: eps, name: join(name, "norm1"))
      |> Axon.activation(non_linearity, name: join(name, "act1"))

    h =
      Axon.conv(h, out_channels,
        kernel_size: 3,
        strides: 1,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv1")
      )

    h =
      if timestep_embedding do
        timestep_embedding
        |> Axon.activation(non_linearity, name: join(name, "timestep.act1"))
        |> Axon.dense(out_channels, name: join(name, "time_emb_proj"))
        |> Axon.nx(&Nx.new_axis(Nx.new_axis(&1, -1), -1))
        |> Axon.add(h)
      else
        h
      end

    h =
      h
      |> Axon.group_norm(groups_out, epsilon: eps, name: join(name, "norm2"))
      |> Axon.activation(non_linearity, name: join(name, "act2"))
      |> Axon.dropout(rate: dropout, name: join(name, "dropout"))
      |> Axon.conv(out_channels,
        kernel_size: 3,
        strides: 1,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv2")
      )

    x =
      if use_nin_shortcut do
        Axon.conv(x, out_channels,
          kernel_size: 1,
          strides: 1,
          padding: :valid,
          name: join(name, "conv_shortcut")
        )
      else
        x
      end

    h
    |> Axon.add(x)
    |> Axon.nx(fn x -> Nx.divide(x, output_scale_factor) end)
  end

  ## Attention Blocks

  @doc """
  Adds a visual attention block to the network.  
  """
  def visual_attention_block(hidden_state, opts) do
    channels = opts[:channels]
    num_groups = opts[:num_groups] || 32
    rescale_output_factor = opts[:rescale_output_factor] || 1.0
    eps = opts[:eps] || 1.0e-5
    name = opts[:name]

    # TODO: Replace me with an actual calculation
    # num_head_channels = opts[:num_head_channels]
    num_heads = 1

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.group_norm(num_groups, epsilon: eps, name: join(name, "group_norm"))
      |> Axon.reshape({channels, :auto})
      |> Axon.transpose([1, 0])

    query =
      hidden_state
      |> Axon.dense(channels, name: join(name, "query"))
      |> transpose_for_scores(num_heads)

    key =
      hidden_state
      |> Axon.dense(channels, name: join(name, "key"))
      |> transpose_for_scores(num_heads)

    value =
      hidden_state
      |> Axon.dense(channels, name: join(name, "value"))
      |> transpose_for_scores(num_heads)

    scale = Nx.divide(1, Nx.sqrt(Nx.sqrt(Nx.divide(channels, num_heads))))

    context_states =
      Axon.layer(
        fn query_states, key_states, value_states, scale, _opts ->
          scaled_query = Nx.multiply(query_states, scale)
          scaled_key = Nx.multiply(Nx.transpose(key_states, axes: [0, 1, 3, 2]), scale)
          attention_scores = Nx.dot(scaled_query, [3], [0, 1], scaled_key, [2], [0, 1])
          attention_probs = Axon.Activations.softmax(attention_scores, axis: -1)

          context_states =
            Nx.dot(
              attention_probs,
              [3],
              [0, 1],
              value_states,
              [2],
              [0, 1]
            )

          context_states = Nx.transpose(context_states, axes: [0, 2, 1, 3])
          {batch, seq, _, _} = Nx.shape(context_states)
          Nx.reshape(context_states, {batch, seq, :auto})
        end,
        [query, key, value, Axon.constant(scale)]
      )

    context_states
    |> Axon.dense(channels, name: join(name, "proj_attn"))
    |> Axon.transpose([1, 0])
    |> then(
      &Axon.layer(
        fn state, residual, _opts ->
          Nx.reshape(state, Nx.shape(residual))
        end,
        [&1, residual]
      )
    )
    |> Axon.add(residual)
    |> Axon.nx(fn x -> Nx.divide(x, rescale_output_factor) end)
  end

  defp transpose_for_scores(input, num_heads) do
    Axon.layer(
      fn input, _opts ->
        {batch, seq, _} = Nx.shape(input)

        input
        |> Nx.reshape({batch, seq, num_heads, :auto})
        |> Nx.transpose(axes: [0, 2, 1, 3])
      end,
      [input]
    )
  end

  @doc """
  Adds a spatial transformer to the network.
  """
  def spatial_transformer(hidden_state, context, opts \\ []) do
    in_channels = opts[:in_channels]
    n_heads = opts[:n_heads]
    d_head = opts[:d_head]
    depth = opts[:depth] || 1
    dropout = opts[:dropout] || 0.0
    context_dim = opts[:context_dim]
    name = opts[:name]

    inner_dim = n_heads * d_head

    residual = hidden_state

    hidden_state
    |> Axon.group_norm(32, epsilon: 1.0e-6, name: join(name, "norm"))
    |> Axon.conv(inner_dim,
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
    |> spatial_transformer_blocks(context,
      dim: inner_dim,
      n_heads: n_heads,
      d_head: d_head,
      dropout: dropout,
      depth: depth,
      context_dim: context_dim,
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
    |> Axon.conv(in_channels,
      kernel_size: 1,
      strides: 1,
      padding: :valid,
      name: join(name, "proj_out")
    )
    |> Axon.add(residual)
  end

  defp spatial_transformer_blocks(hidden_state, context, opts) do
    name = opts[:name]

    for idx <- 0..(opts[:depth] - 1), reduce: hidden_state do
      hidden_state ->
        basic_transformer_block(hidden_state, context,
          dim: opts[:dim],
          context_dim: opts[:context_dim],
          n_heads: opts[:n_heads],
          d_head: opts[:d_head],
          dropout: opts[:dropout],
          name: join(name, "transformer_blocks.#{idx}")
        )
    end
  end

  defp basic_transformer_block(hidden_state, context, opts) do
    name = opts[:name]

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(name: join(name, "norm1"), channel_index: 2)
      |> cross_attention(nil,
        query_dim: opts[:dim],
        heads: opts[:n_heads],
        dim_head: opts[:d_head],
        dropout: opts[:dropout],
        name: join(name, "attn1")
      )
      |> Axon.add(residual)

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(name: join(name, "norm2"), channel_index: 2)
      |> cross_attention(context,
        query_dim: opts[:dim],
        context_dim: opts[:context_dim],
        heads: opts[:n_heads],
        dim_head: opts[:d_head],
        dropout: opts[:dropout],
        name: join(name, "attn2")
      )
      |> Axon.add(residual)

    residual = hidden_state

    hidden_state
    |> Axon.layer_norm(name: join(name, "norm3"), channel_index: 2)
    |> feed_forward(
      dim: opts[:dim],
      dropout: opts[:dropout],
      name: join(name, "ff")
    )
    |> Axon.add(residual)
  end

  defp cross_attention(hidden_state, context, opts) do
    name = opts[:name]
    # TODO: work these in?
    query_dim = opts[:query_dim]
    # context_dim = opts[:context_dim]
    heads = opts[:heads] || 8
    dim_head = opts[:dim_head] || 64
    dropout = opts[:dropout] || 0.0

    inner_dim = dim_head * heads

    context = if context, do: context, else: hidden_state

    query =
      hidden_state
      |> Axon.dense(inner_dim, use_bias: false, name: join(name, "to_q"))
      |> Layers.split_heads(heads)

    key =
      context
      |> Axon.dense(inner_dim, use_bias: false, name: join(name, "to_k"))
      |> Layers.split_heads(heads)

    value =
      context
      |> Axon.dense(inner_dim, use_bias: false, name: join(name, "to_v"))
      |> Layers.split_heads(heads)

    # TODO: Attention mask
    attention_weights =
      Layers.attention_weights(query, key, Axon.constant(Nx.tensor(0)))
      |> Axon.dropout(rate: dropout, name: join(name, "dropout"))

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()

    attention_output
    |> Axon.dense(query_dim, name: join(name, "to_out.0"))
    |> Axon.dropout(rate: dropout)
  end

  defp feed_forward(x, opts) do
    dim = opts[:dim]
    dim_out = opts[:dim_out] || dim
    mult = opts[:mult] || 4
    dropout = opts[:dropout] || 0.0
    name = opts[:name]

    inner_dim = mult * dim

    x
    |> geglu(dim_in: dim, dim_out: inner_dim, name: join(name, "net.0"))
    |> Axon.dropout(rate: dropout)
    |> Axon.dense(dim_out, name: join(name, "net.2"))
  end

  defp geglu(x, opts) do
    dim_in = opts[:dim_in]
    dim_out = opts[:dim_out]
    name = opts[:name]

    {x, gate} =
      x
      |> Axon.dense(dim_out * 2, name: join(name, "proj"))
      |> Axon.split(2, axis: -1)

    Axon.multiply(x, Axon.gelu(gate))
  end
end
