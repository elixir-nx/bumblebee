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

    * `:base` - the base for the frequency exponentiation. Defaults
      to `10_000`.

    * `:scale` - a multiplier for the sin/cos arguments. Defaults to `1`

  """
  def timestep_sinusoidal_embedding(timestep, embedding_size, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        flip_sin_to_cos: false,
        frequency_correction_term: 1,
        base: 10_000,
        scale: 1
      )

    Axon.layer(&timestep_sinusoidal_embedding_impl/2, [timestep],
      embedding_size: embedding_size,
      op_name: :timestep_sinusoidal_embedding,
      flip_sin_to_cos: opts[:flip_sin_to_cos],
      frequency_correction_term: opts[:frequency_correction_term],
      scale: opts[:scale],
      base: opts[:base]
    )
  end

  defnp timestep_sinusoidal_embedding_impl(timestep, opts \\ []) do
    opts =
      keyword!(opts, [
        :embedding_size,
        flip_sin_to_cos: false,
        frequency_correction_term: 1,
        scale: 1,
        base: 10_000,
        mode: :train
      ])

    embedding_size = opts[:embedding_size]
    base = opts[:base]
    frequency_correction_term = opts[:frequency_correction_term]

    if rem(embedding_size, 2) != 0 do
      raise ArgumentError,
            "expected embedding size to an even number, but got: #{inspect(embedding_size)}"
    end

    half_size = div(embedding_size, 2)

    frequency =
      Nx.exp(-Nx.log(base) * Nx.iota({half_size}) / (half_size - frequency_correction_term))

    angle = Nx.new_axis(timestep, -1) * Nx.new_axis(frequency, 0)
    angle = opts[:scale] * angle

    if opts[:flip_sin_to_cos] do
      Nx.concatenate([Nx.cos(angle), Nx.sin(angle)], axis: -1)
    else
      Nx.concatenate([Nx.sin(angle), Nx.cos(angle)], axis: -1)
    end
  end

  @doc """
  Adds a residual block to the network.
  """
  def residual_block(hidden_state, in_channels, out_channels, opts \\ []) do
    timestep_embedding = opts[:timestep_embedding]
    dropout = opts[:dropout] || 0.0
    norm_num_groups = opts[:norm_num_groups] || 32
    norm_num_groups_out = opts[:norm_num_groups_out] || norm_num_groups
    norm_epsilon = opts[:norm_epsilon] || 1.0e-6
    activation = opts[:activation] || :swish
    output_scale_factor = opts[:output_scale_factor] || 1.0
    project_shortcut? = Keyword.get(opts, :project_shortcut?, in_channels != out_channels)
    name = opts[:name]

    shortcut =
      if project_shortcut? do
        Axon.conv(hidden_state, out_channels,
          kernel_size: 1,
          strides: 1,
          padding: :valid,
          name: join(name, "shortcut.projection")
        )
      else
        hidden_state
      end

    hidden_state =
      hidden_state
      |> Axon.group_norm(norm_num_groups, epsilon: norm_epsilon, name: join(name, "norm_1"))
      |> Axon.activation(activation)
      |> Axon.conv(out_channels,
        kernel_size: 3,
        strides: 1,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv_1")
      )

    hidden_state =
      if timestep_embedding do
        timestep_embedding
        |> Axon.activation(activation)
        |> Axon.dense(out_channels, name: join(name, "timestep_projection"))
        |> Axon.nx(&Nx.new_axis(Nx.new_axis(&1, 1), 1))
        |> Axon.add(hidden_state)
      else
        hidden_state
      end

    hidden_state =
      hidden_state
      |> Axon.group_norm(norm_num_groups_out, epsilon: norm_epsilon, name: join(name, "norm_2"))
      |> Axon.activation(activation)
      |> Axon.dropout(rate: dropout)
      |> Axon.conv(out_channels,
        kernel_size: 3,
        strides: 1,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv_2")
      )

    hidden_state
    |> Axon.add(shortcut)
    |> Axon.nx(&Nx.divide(&1, output_scale_factor))
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
            Axon.nx(hidden_state, &Nx.pad(&1, 0.0, [{0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 0}]))

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
  Adds an upsample block to the network.
  """
  def upsample_2d(hidden_state, channels, opts \\ []) do
    name = opts[:name]

    hidden_state
    |> Axon.nx(fn hidden_state ->
      {_, h, w, _} = Nx.shape(hidden_state)
      Axon.Layers.resize(hidden_state, size: {2 * h, 2 * w}, mode: :nearest)
    end)
    |> Axon.conv(channels,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv")
    )
  end
end
