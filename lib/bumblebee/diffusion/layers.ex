defmodule Bumblebee.Diffusion.Layers do
  @moduledoc false

  import Bumblebee.Utils.Model, only: [join: 2]
  import Nx.Defn

  @doc """
  Adds a sinusoidal embedding layer to the network.

  Expects a batched timestep tensor (1-dimensional).

  ## Options

    * `:flip_sin_to_cos` - whether to swap the order of sine and cosine
      order in the embedding. Defaults to `false`

    * `:frequency_correction_term` - controls the frequency formula used.
      The frequency is computed as
      $\omega_i = \frac{1}{10000^{\frac{i}{n - s}}}$, for $i \in \{0, ..., n-1\}$,
      where $n$ is half of the embedding size and $s$ is the shift.
      Historically, certain implementations of sinusoidal embedding
      used $s=0$, while other used $s=1$. Defaults to `0`

    * `:max_period` - the base for the frequency exponentiation. Defaults
      to `10_000`.

    * `:scale` - a multiplier for the sin/cos arguments. Defaults to `1`

  """
  def timestep_sinusoidal_embedding(timestep, embedding_size, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        flip_sin_to_cos: false,
        frequency_correction_term: 1,
        max_period: 10_000,
        scale: 1
      )

    Axon.layer(&timestep_sinusoidal_embedding_impl/2, [timestep],
      embedding_size: embedding_size,
      op_name: :timestep_sinusoidal_embedding,
      flip_sin_to_cos: opts[:flip_sin_to_cos],
      frequency_correction_term: opts[:frequency_correction_term],
      scale: opts[:scale],
      max_period: opts[:max_period]
    )
  end

  defnp timestep_sinusoidal_embedding_impl(timestep, opts \\ []) do
    opts =
      keyword!(opts, [
        :embedding_size,
        flip_sin_to_cos: false,
        frequency_correction_term: 1,
        scale: 1,
        max_period: 10_000,
        mode: :train
      ])

    embedding_size = opts[:embedding_size]
    max_period = opts[:max_period]
    frequency_correction_term = opts[:frequency_correction_term]

    if rem(embedding_size, 2) != 0 do
      raise ArgumentError,
            "expected embedding size to an even number, but got: #{inspect(embedding_size)}"
    end

    half_size = div(embedding_size, 2)

    frequency =
      Nx.exp(-Nx.log(max_period) * Nx.iota({half_size}) / (half_size - frequency_correction_term))

    angle = Nx.new_axis(timestep, -1) * Nx.new_axis(frequency, 0)
    angle = opts[:scale] * angle

    if opts[:flip_sin_to_cos] do
      Nx.concatenate([Nx.cos(angle), Nx.sin(angle)], axis: -1)
    else
      Nx.concatenate([Nx.sin(angle), Nx.cos(angle)], axis: -1)
    end
  end

  @doc """
  Adds a ResNet block to the network.
  """
  def resnet_block(x, in_channels, out_channels, opts \\ []) do
    timestep_embeds = opts[:timestep_embeds]
    dropout = opts[:dropout] || 0.0
    num_groups = opts[:num_groups] || 32
    num_groups_out = opts[:num_groups_out] || num_groups
    epsilon = opts[:epsilon] || 1.0e-6
    activation = opts[:activation] || :swish
    output_scale_factor = opts[:output_scale_factor] || 1.0
    use_shortcut = Keyword.get(opts, :use_shortcut, in_channels != out_channels)
    name = opts[:name]

    h = x

    h =
      h
      |> Axon.group_norm(num_groups, epsilon: epsilon, name: join(name, "norm1"))
      |> Axon.activation(activation, name: join(name, "act1"))

    h =
      Axon.conv(h, out_channels,
        kernel_size: 3,
        strides: 1,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv1")
      )

    h =
      if timestep_embeds do
        timestep_embeds
        |> Axon.activation(activation, name: join(name, "timestep.act1"))
        |> Axon.dense(out_channels, name: join(name, "time_emb_proj"))
        |> Axon.nx(&Nx.new_axis(Nx.new_axis(&1, -1), -1))
        |> Axon.add(h)
      else
        h
      end

    h =
      h
      |> Axon.group_norm(num_groups_out, epsilon: epsilon, name: join(name, "norm2"))
      |> Axon.activation(activation, name: join(name, "act2"))
      |> Axon.dropout(rate: dropout, name: join(name, "dropout"))
      |> Axon.conv(out_channels,
        kernel_size: 3,
        strides: 1,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv2")
      )

    x =
      if use_shortcut do
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

  @doc """
  Adds a downsample block to the network.
  """
  def downsample_2d(hidden_state, channels, opts \\ []) do
    padding = opts[:padding]
    name = opts[:name]
    stride = 2

    {hidden_state, padding} =
      cond do
        padding == 0 ->
          hidden_state =
            Axon.nx(hidden_state, &Nx.pad(&1, 0.0, [{0, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 1, 0}]))

          {hidden_state, :valid}

        padding == 1 ->
          {hidden_state, [{1, 1}, {1, 1}]}

        true ->
          {hidden_state, padding}
      end

    Axon.conv(hidden_state, channels,
      kernel_size: 3,
      strides: stride,
      padding: padding,
      name: join(name, "conv")
    )
  end

  @doc """
  Adds an upsample layer to the network.
  """
  def upsample_2d(hidden_state, channels, opts \\ []) do
    name = opts[:name]

    hidden_state
    |> Axon.nx(fn hidden_state ->
      {_, _, h, w} = Nx.shape(hidden_state)
      Axon.Layers.resize(hidden_state, size: {2 * h, 2 * w}, mode: :nearest)
    end)
    |> Axon.conv(channels,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv")
    )
  end
end
