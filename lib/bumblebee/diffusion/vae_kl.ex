defmodule Bumblebee.Diffusion.VaeKl do
  @moduledoc """
  Variational autoencoder (VAE) with Kullback–Leibler divergence (KL) loss.

  ## Architectures

    * `:base` - the entire VAE model

    * `:encoder` - just the encoder part of the base model

    * `:decoder` - just the decoder part of the base model

  ## Inputs

    * `"sample"` - `{batch_size, in_channels, sample_size, sample_size}`

      Sample input with two spatial dimensions. Note that in case of
      the `:decoder` model, the input usually has lower dimensionality.

    * `"sample_posterior"` - `{}`

      When `true`, the decoder input is sampled from the encoder output
      distribution. Otherwise the distribution mode value is used instead.
      This input is only relevant for the `:base` model. Defaults to `false`.

  ## Config

    * `:in_channels` - the number of channels in the input. Defaults
      to `3`

    * `:out_channels` - the number of channels in the output. Defaults
      to `3`

    * `:sample_size` - size of the input spatial dimensions. Defaults
      to `32`

    * `:latent_channels` - the number of channels in the latent space.
      Defaults to `4`

    * `:down_block_types`- a list of downsample block types. Currently
      the only supported type is `:down_block`. Defaults to `[:down_block]`

    * `:up_block_types`- a list of upsampling block types. Currently
      the only supported type is `:up_block`. Defaults to `[:up_block]`

    * `:block_out_channels` - a list of block output channels. Defaults
      to `[64]`

    * `:layers_per_block` - the number of ResNet layers in each block.
      Defaults to `1`

    * `:act_fn` - the activation function. Defaults to `:silu`

  ## References

    * [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers
  alias Bumblebee.Diffusion

  defstruct architecture: :base,
            in_channels: 3,
            out_channels: 3,
            sample_size: 32,
            latent_channels: 4,
            down_block_types: [:down_block],
            up_block_types: [:up_block],
            block_out_channels: [64],
            layers_per_block: 1,
            act_fn: :silu

  @behaviour Bumblebee.ModelSpec

  @impl true
  def architectures(), do: [:base, :encoder, :decoder]

  @impl true
  def base_model_prefix(), do: "vae"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def input_template(config) do
    sample_shape = config |> sample_shape() |> put_elem(0, 1)
    %{"sample" => Nx.template(sample_shape, :f32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs = inputs(config)
    sample = vae_kl(inputs, config)
    Layers.output(%{sample: sample})
  end

  def model(%__MODULE__{architecture: :encoder} = config) do
    inputs = inputs(config)
    posterior = encode(inputs["sample"], config)
    Layers.output(%{latent_dist: Axon.container(posterior)})
  end

  def model(%__MODULE__{architecture: :decoder} = config) do
    inputs = inputs(config)
    sample = decode(inputs["sample"], config)
    Layers.output(%{sample: sample})
  end

  defp inputs(%__MODULE__{architecture: :base} = config) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("sample", shape: sample_shape(config)),
      Axon.input("sample_posterior", shape: {}, optional: true)
    ])
  end

  defp inputs(config) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("sample", shape: sample_shape(config))
    ])
  end

  defp sample_shape(%__MODULE__{architecture: :decoder} = config) do
    downsample_rate = (length(config.block_out_channels) - 1) * 2
    size = div(config.sample_size, downsample_rate)
    {nil, config.latent_channels, size, size}
  end

  defp sample_shape(config) do
    {nil, config.in_channels, config.sample_size, config.sample_size}
  end

  defp vae_kl(inputs, config, opts \\ []) do
    name = opts[:name]

    x = inputs["sample"]

    sample_posterior =
      Layers.default inputs["sample_posterior"] do
        Axon.constant(Nx.tensor(0, type: {:u, 8}))
      end

    posterior = encode(x, config, name: name)

    z =
      Axon.cond(
        sample_posterior,
        &Nx.equal(&1, Nx.tensor(1)),
        sample(posterior),
        mode(posterior)
      )

    decode(z, config, name: name)
  end

  defp encode(x, config, opts \\ []) do
    name = opts[:name]

    x
    |> encoder(config, name: join(name, "encoder"))
    |> Axon.conv(2 * config.latent_channels, kernel_size: 1, name: join(name, "quant_conv"))
    |> diagonal_gaussian_distribution()
  end

  defp decode(z, config, opts \\ []) do
    name = opts[:name]

    z
    |> Axon.conv(config.latent_channels, kernel_size: 1, name: join(name, "post_quant_conv"))
    |> decoder(config, name: join(name, "decoder"))
  end

  defp encoder(x, config, opts) do
    name = opts[:name]

    x
    |> Axon.conv(hd(config.block_out_channels),
      kernel_size: 3,
      strides: 1,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv_in")
    )
    |> down_blocks(config, name: join(name, "down_blocks"))
    |> mid_block(config, name: join(name, "mid_block"))
    |> Axon.group_norm(32, epsilon: 1.0e-6, name: join(name, "conv_norm_out"))
    |> Axon.activation(:silu, name: join(name, "activation"))
    |> Axon.conv(2 * config.latent_channels,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv_out")
    )
  end

  defp decoder(z, config, opts) do
    name = opts[:name]

    z
    |> Axon.conv(List.last(config.block_out_channels),
      kernel_size: 3,
      strides: 1,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv_in")
    )
    |> mid_block(config, name: join(name, "mid_block"))
    |> up_blocks(config, name: join(name, "up_blocks"))
    |> Axon.group_norm(32, epsilon: 1.0e-6, name: join(name, "conv_norm_out"))
    |> Axon.activation(:silu, name: join(name, "activation"))
    |> Axon.conv(config.out_channels,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv_out")
    )
  end

  defp down_blocks(sample, config, opts) do
    name = opts[:name]
    blocks = Enum.zip(config.block_out_channels, config.down_block_types)

    acc = {sample, hd(config.block_out_channels)}

    {sample, _} =
      for {{output_channel, down_block_type}, idx} <- Enum.with_index(blocks), reduce: acc do
        {sample, in_channels} ->
          last_block? = idx == length(config.block_out_channels) - 1

          block_opts = [
            num_layers: config.layers_per_block,
            in_channels: in_channels,
            out_channels: output_channel,
            add_downsample: not last_block?,
            resnet_activation: config.act_fn,
            name: join(name, idx)
          ]

          {do_down_block(down_block_type, sample, block_opts), output_channel}
      end

    sample
  end

  defp do_down_block(:down_block, sample, opts),
    do: down_block(sample, opts)

  defp down_block(hidden_state, opts) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    num_layers = opts[:num_layers]
    resnet_activation = opts[:resnet_activation]
    add_downsample = opts[:add_downsample]
    name = opts[:name]

    hidden_state =
      for idx <- 0..(num_layers - 1), reduce: hidden_state do
        hidden_state ->
          in_channels = if(idx == 0, do: in_channels, else: out_channels)

          Diffusion.Layers.resnet_block(hidden_state, in_channels, out_channels,
            activation: resnet_activation,
            name: join(name, "resnets.#{idx}")
          )
      end

    if add_downsample do
      Diffusion.Layers.downsample_2d(hidden_state, out_channels,
        padding: 0,
        name: join(name, "downsamplers.0")
      )
    else
      hidden_state
    end
  end

  defp up_blocks(sample, config, opts) do
    name = opts[:name]
    reversed_block_out_channels = Enum.reverse(config.block_out_channels)
    blocks = Enum.zip(reversed_block_out_channels, config.up_block_types)

    acc = {sample, hd(reversed_block_out_channels)}

    {sample, _} =
      for {{output_channel, up_block_type}, idx} <- Enum.with_index(blocks), reduce: acc do
        {sample, in_channels} ->
          last_block? = idx == length(config.block_out_channels) - 1

          block_opts = [
            num_layers: config.layers_per_block + 1,
            in_channels: in_channels,
            out_channels: output_channel,
            add_upsample: not last_block?,
            resnet_activation: config.act_fn,
            name: join(name, idx)
          ]

          {do_up_block(up_block_type, sample, block_opts), output_channel}
      end

    sample
  end

  defp do_up_block(:up_block, sample, opts),
    do: up_block(sample, opts)

  defp up_block(hidden_state, opts) do
    in_channels = opts[:in_channels]
    out_channels = opts[:out_channels]
    num_layers = opts[:num_layers]
    resnet_activation = opts[:resnet_activation]
    add_upsample = opts[:add_upsample]
    name = opts[:name]

    hidden_state =
      for idx <- 0..(num_layers - 1), reduce: hidden_state do
        hidden_state ->
          in_channels = if(idx == 0, do: in_channels, else: out_channels)

          Diffusion.Layers.resnet_block(hidden_state, in_channels, out_channels,
            activation: resnet_activation,
            name: join(name, "resnets.#{idx}")
          )
      end

    if add_upsample do
      Diffusion.Layers.upsample_2d(hidden_state, out_channels, name: join(name, "upsamplers.0"))
    else
      hidden_state
    end
  end

  defp mid_block(hidden_state, config, opts) do
    name = opts[:name]

    in_channels = List.last(config.block_out_channels)

    hidden_state
    |> Diffusion.Layers.resnet_block(in_channels, in_channels,
      activation: config.act_fn,
      name: join(name, "resnets.0")
    )
    |> visual_attention(in_channels, num_heads: 1, name: join(name, "attentions.0"))
    |> Diffusion.Layers.resnet_block(in_channels, in_channels,
      activation: config.act_fn,
      name: join(name, "resnets.1")
    )
  end

  defp visual_attention(hidden_state, channels, opts) do
    num_heads = opts[:num_heads]
    name = opts[:name]

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.group_norm(32, epsilon: 1.0e-6, name: join(name, "group_norm"))
      |> Axon.reshape({channels, :auto})
      |> Axon.transpose([1, 0])

    query =
      hidden_state
      |> Axon.dense(channels, name: join(name, "query"))
      |> Layers.split_heads(num_heads)

    key =
      hidden_state
      |> Axon.dense(channels, name: join(name, "key"))
      |> Layers.split_heads(num_heads)

    value =
      hidden_state
      |> Axon.dense(channels, name: join(name, "value"))
      |> Layers.split_heads(num_heads)

    attention_weights = Layers.attention_weights(query, key, Axon.constant(Nx.tensor(0)))

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()

    attention_output
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
  end

  defp diagonal_gaussian_distribution(x) do
    {mean, logvar} = Axon.split(x, 2, axis: 1)
    logvar = Axon.nx(logvar, &Nx.clip(&1, -30.0, 20.0))
    std = Axon.nx(logvar, &Nx.exp(Nx.multiply(0.5, &1)))
    var = Axon.nx(logvar, &Nx.exp/1)
    %{mean: mean, logvar: logvar, std: std, var: var}
  end

  defp sample(posterior) do
    z = Axon.nx(posterior.mean, &Nx.random_normal(Nx.shape(&1)))

    posterior.mean
    |> Axon.add(posterior.std)
    |> Axon.multiply(z)
  end

  defp mode(%{mean: mean}) do
    mean
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["act_fn"])
      |> Shared.convert_common()
      |> Shared.map_items("down_block_types", %{"DownEncoderBlock2D" => :down_block})
      |> Shared.map_items("up_block_types", %{"UpDecoderBlock2D" => :up_block})
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
