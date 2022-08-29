defmodule Bumblebee.Diffusion.AutoencoderKl do
  @moduledoc """
  AutoencoderKL.
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers
  alias Bumblebee.Diffusion

  defstruct architecture: :base,
            in_channels: 3,
            out_channels: 3,
            height: 512,
            width: 512,
            down_block_types: [:down_encoder_block_2d],
            up_block_types: [:up_decoder_block_2d],
            block_out_channels: [64],
            layers_per_block: 1,
            act_fn: :silu,
            latent_channels: 4,
            sample_size: 32

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
  def input_template(%__MODULE__{architecture: :decoder} = config) do
    sample_height = div(config.height, 16)
    sample_width = div(config.width, 16)
    sample_shape = {1, config.latent_channels, sample_height, sample_width}

    %{"sample" => Nx.template(sample_shape, :f32)}
  end

  def input_template(config) do
    sample_height = div(config.height, 8)
    sample_width = div(config.width, 8)
    sample_shape = {1, config.in_channels, sample_height, sample_width}

    %{
      "sample" => Nx.template(sample_shape, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :decoder} = config) do
    inputs = inputs(config)
    decode(inputs["sample"], config, name: "decoder")
  end

  def model(%__MODULE__{architecture: :base} = config) do
    inputs = inputs(config)
    autoencoder_kl(inputs, config, name: "vae")
  end

  defp inputs(config) do
    sample_height = div(config.height, 8)
    sample_width = div(config.width, 8)
    sample_shape = {nil, config.in_channels, sample_height, sample_width}

    %{
      "sample" => Axon.input("sample", shape: sample_shape),
      "sample_posterior" => Axon.input("sample_posterior", shape: {}, optional: true)
    }
  end

  defp autoencoder_kl(inputs, config, opts) do
    name = opts[:name]

    x = inputs["sample"]

    sample_posterior =
      Layers.default inputs["sample_posterior"] do
        Axon.constant(Nx.tensor(0, type: {:u, 8}))
      end

    posterior = encode(x, config, name: join(name, "encoder"))

    z =
      Axon.cond(
        sample_posterior,
        &Nx.equal(&1, Nx.tensor(1)),
        sample(posterior),
        mode(posterior)
      )

    decode(z, config, name: join(name, "decoder"))
  end

  defp encode(x, config, opts) do
    name = opts[:name]

    x
    |> encoder(config, name: name)
    |> Axon.conv(2 * config.latent_channels, kernel_size: 1, name: "quant_conv")
    |> diagonal_gaussian_distribution(config)
  end

  defp decode(z, config, opts) do
    name = opts[:name]

    z
    |> Axon.conv(config.latent_channels, kernel_size: 1, name: "post_quant_conv")
    |> decoder(config, name: name)
  end

  defp encoder(x, config, opts) do
    name = opts[:name]

    x
    |> Axon.conv(Enum.at(config.block_out_channels, 0),
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
    |> Axon.conv(Enum.at(config.block_out_channels, Enum.count(config.block_out_channels) - 1),
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

    acc = {sample, Enum.at(config.block_out_channels, 0)}

    {sample, _} =
      for {{output_channel, down_block_type}, idx} <- Enum.with_index(blocks), reduce: acc do
        {sample, in_channels} ->
          block_opts = [
            num_layers: config.layers_per_block,
            in_channels: in_channels,
            out_channels: output_channel,
            add_downsample: idx != Enum.count(config.block_out_channels) - 1,
            resnet_eps: 1.0e-6,
            downsample_padding: 0,
            resnet_act_fn: config.act_fn,
            attn_num_head_channels: nil,
            temb_channels: nil,
            name: join(name, "#{idx}")
          ]

          {Diffusion.Layers.apply_unet_block(down_block_type, sample, block_opts), output_channel}
      end

    sample
  end

  defp up_blocks(sample, config, opts) do
    name = opts[:name]
    reversed_block_out_channels = Enum.reverse(config.block_out_channels)
    blocks = Enum.zip(reversed_block_out_channels, config.up_block_types)

    acc = {sample, Enum.at(reversed_block_out_channels, 0)}

    {sample, _} =
      for {{output_channel, up_block_type}, idx} <- Enum.with_index(blocks), reduce: acc do
        {sample, in_channels} ->
          block_opts = [
            num_layers: config.layers_per_block + 1,
            in_channels: in_channels,
            out_channels: output_channel,
            prev_output_channel: nil,
            add_upsample: idx != Enum.count(config.block_out_channels) - 1,
            resnet_eps: 1.0e-6,
            resnet_act_fn: config.act_fn,
            attn_num_head_channels: nil,
            name: join(name, "#{idx}")
          ]

          {Diffusion.Layers.apply_unet_block(up_block_type, sample, block_opts), output_channel}
      end

    sample
  end

  defp mid_block(sample, config, opts) do
    name = opts[:name]

    Diffusion.Layers.mid_block_2d(sample,
      in_channels: Enum.at(config.block_out_channels, Enum.count(config.block_out_channels) - 1),
      resnet_eps: 1.0e-6,
      resnet_act_fn: config.act_fn,
      output_scale_factor: 1,
      resnet_time_scale_shift: :default,
      attn_num_head_channels: nil,
      resnet_groups: 32,
      temb_channels: nil,
      name: name
    )
  end

  defp diagonal_gaussian_distribution(x, _config) do
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
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
