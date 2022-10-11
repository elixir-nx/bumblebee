defmodule Bumblebee.Diffusion.UNet2DConditional do
  @moduledoc ~S"""
  U-Net model with two spatial dimensions and conditional state.

  ## Architectures

    * `:base` - the U-Net model

  ## Inputs

    * `"sample"` - `{batch_size, in_channels, sample_size, sample_size}`

      Sample input with two spatial dimensions.

    * `"timestep"` - `{}`

      The timestep used to parameterize model behaviour in a multi-step
      process, such as diffusion.

    * `"encoder_last_hidden_state"` - `{batch_size, seq_length, hidden_size}`

      The conditional state (context) to use with cross-attention.

  ## Config

    * `:in_channels` - the number of channels in the input. Defaults
      to `4`

    * `:out_channels` - the number of channels in the output. Defaults
      to `4`

    * `:sample_size` - size of the input spatial dimensions. Defaults
      to `32`

    * `:center_input_sample` - whether to center the input sample.
      Defaults to `false`

    * `:flip_sin_to_cos` - whether to flip the sin to cos in the sinusoidal
      timestep embedding. Defaults to `true`

    * `:freq_shift` - controls the frequency formula in the timestep
      sinusoidal embedding. The frequency is computed as
      $\omega_i = \frac{1}{10000^{\frac{i}{n - s}}}$, for $i \in \{0, ..., n-1\}$,
      where $n$ is half of the embedding size and $s$ is the shift.
      Historically, certain implementations of sinusoidal embedding
      used $s=0$, while other used $s=1$. Defaults to `0`

    * `:down_block_types`- a list of downsample block types. The supported
      blocks are: `:down_block`, `:cross_attention_down_block`. Defaults to
      `[:cross_attention_down_block, :cross_attention_down_block, :cross_attention_down_block, :down_block]`

    * `:up_block_types`- a list of upsampling block types. The supported
      blocks are: `:up_block`, `:cross_attention_up_block`, Defaults to
      `[:up_block, :cross_attention_up_block, :cross_attention_up_block, :cross_attention_up_block]`

    * `:block_out_channels` - a list of block output channels. Defaults
      to `[320, 640, 1280, 1280]`

    * `:layers_per_block` - the number of ResNet layers in each block.
      Defaults to `2`

    * `:downsample_padding` - the padding to use in the downsampling
      convolution. Defaults to `[{1, 1}, {1, 1}]`

    * `:mid_block_scale_factor` - the scale factor to use for the mid
      block. Defaults to `1`

    * `:act_fn` - the activation function. Defaults to `:silu`

    * `:norm_num_groups` - the number of groups to use for normalization.
      Defaults to `32`

    * `:norm_eps` - the epsilon to use for normalization. Defaults to
      `1.0e-5`

    * `:cross_attention_dim` - dimensionality of the cross attention
      features. Defaults to `1280`

    * `:attention_head_dim` - the number of attention heads. Defaults
      to `8`

  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers
  alias Bumblebee.Diffusion

  defstruct architecture: :base,
            in_channels: 4,
            out_channels: 4,
            sample_size: 32,
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0,
            down_block_types: [
              :cross_attention_down_block,
              :cross_attention_down_block,
              :cross_attention_down_block,
              :down_block
            ],
            up_block_types: [
              :up_block,
              :cross_attention_up_block,
              :cross_attention_up_block,
              :cross_attention_up_block
            ],
            block_out_channels: [320, 640, 1280, 1280],
            layers_per_block: 2,
            downsample_padding: [{1, 1}, {1, 1}],
            mid_block_scale_factor: 1,
            act_fn: :silu,
            norm_num_groups: 32,
            norm_eps: 1.0e-5,
            cross_attention_dim: 1280,
            attention_head_dim: 8

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures(), do: [:base]

  @impl true
  def base_model_prefix(), do: "unet"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def input_template(config) do
    sample_shape = {1, config.in_channels, config.sample_size, config.sample_size}
    timestep_shape = {}
    encoder_last_hidden_state_shape = {1, 1, config.cross_attention_dim}

    %{
      "sample" => Nx.template(sample_shape, :f32),
      "timestep" => Nx.template(timestep_shape, :s64),
      "encoder_last_hidden_state" => Nx.template(encoder_last_hidden_state_shape, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs = inputs(config)
    sample = unet_2d_conditional(inputs, config)
    Layers.output(%{sample: sample})
  end

  defp inputs(config) do
    sample_shape = {nil, config.in_channels, config.sample_size, config.sample_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("sample", shape: sample_shape),
      Axon.input("timestep", shape: {}),
      Axon.input("encoder_last_hidden_state", shape: {nil, nil, config.cross_attention_dim})
    ])
  end

  defp unet_2d_conditional(inputs, config, opts \\ []) do
    name = opts[:name]

    sample = inputs["sample"]
    timestep = inputs["timestep"]
    encoder_last_hidden_state = inputs["encoder_last_hidden_state"]

    sample =
      if config.center_input_sample do
        Axon.nx(sample, fn sample -> 2 * sample - 1.0 end, op_name: :center)
      else
        sample
      end

    timestep =
      Axon.layer(
        fn sample, timestep, _opts ->
          Nx.broadcast(timestep, {Nx.axis_size(sample, 0)})
        end,
        [sample, timestep],
        op_name: :broadcast
      )

    timestep_embeds =
      timestep
      |> Diffusion.Layers.timestep_sinusoidal_embedding(hd(config.block_out_channels),
        flip_sin_to_cos: config.flip_sin_to_cos,
        frequency_correction_term: config.freq_shift
      )
      |> Diffusion.Layers.UNet.timestep_embedding_mlp(hd(config.block_out_channels) * 4,
        name: "time_embedding"
      )

    sample =
      Axon.conv(sample, hd(config.block_out_channels),
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv_in")
      )

    {sample, down_block_residuals} =
      down_blocks(sample, timestep_embeds, encoder_last_hidden_state, config,
        name: join(name, "down_blocks")
      )

    sample
    |> mid_block(timestep_embeds, encoder_last_hidden_state, config, name: join(name, "mid_block"))
    |> up_blocks(timestep_embeds, down_block_residuals, encoder_last_hidden_state, config,
      name: join(name, "up_blocks")
    )
    |> Axon.group_norm(config.norm_num_groups,
      epsilon: config.norm_eps,
      name: join(name, "conv_norm_out")
    )
    |> Axon.activation(:silu, name: join(name, "conv_act"))
    |> Axon.conv(config.out_channels,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv_out")
    )
  end

  defp down_blocks(sample, timestep_embeds, encoder_last_hidden_state, config, opts) do
    name = opts[:name]
    blocks = Enum.zip(config.block_out_channels, config.down_block_types)

    in_channels = hd(config.block_out_channels)
    down_block_residuals = [{sample, in_channels}]

    state = {sample, down_block_residuals, in_channels}

    {sample, down_block_residuals, _} =
      for {{out_channels, block_type}, idx} <- Enum.with_index(blocks), reduce: state do
        {sample, down_block_residuals, in_channels} ->
          last_block? = idx == length(config.block_out_channels) - 1

          {sample, residuals} =
            Diffusion.Layers.UNet.down_block_2d(
              block_type,
              sample,
              timestep_embeds,
              encoder_last_hidden_state,
              num_layers: config.layers_per_block,
              in_channels: in_channels,
              out_channels: out_channels,
              add_downsample: not last_block?,
              downsample_padding: config.downsample_padding,
              resnet_activation: config.act_fn,
              resnet_epsilon: config.norm_eps,
              resnet_num_groups: config.norm_num_groups,
              num_attention_heads: config.attention_head_dim,
              name: join(name, idx)
            )

          {sample, down_block_residuals ++ Tuple.to_list(residuals), out_channels}
      end

    {sample, List.to_tuple(down_block_residuals)}
  end

  defp mid_block(hidden_state, timesteps_embedding, encoder_last_hidden_state, config, opts) do
    Diffusion.Layers.UNet.mid_cross_attention_block_2d(
      hidden_state,
      timesteps_embedding,
      encoder_last_hidden_state,
      channels: List.last(config.block_out_channels),
      resnet_activation: config.act_fn,
      resnet_epsilon: config.norm_eps,
      resnet_num_groups: config.norm_num_groups,
      output_scale_factor: config.mid_block_scale_factor,
      num_attention_heads: config.attention_head_dim,
      name: opts[:name]
    )
  end

  defp up_blocks(
         sample,
         timestep_embeds,
         down_block_residuals,
         encoder_last_hidden_state,
         config,
         opts
       ) do
    name = opts[:name]

    down_block_residuals =
      down_block_residuals
      |> Tuple.to_list()
      |> Enum.reverse()
      |> Enum.chunk_every(config.layers_per_block + 1)

    reversed_block_out_channels = Enum.reverse(config.block_out_channels)
    in_channels = hd(reversed_block_out_channels)

    blocks_and_chunks =
      [reversed_block_out_channels, config.up_block_types, down_block_residuals]
      |> Enum.zip()
      |> Enum.with_index()

    {sample, _} =
      for {{out_channels, block_type, residuals}, idx} <- blocks_and_chunks,
          reduce: {sample, in_channels} do
        {sample, in_channels} ->
          last_block? = idx == length(config.block_out_channels) - 1

          sample =
            Diffusion.Layers.UNet.up_block_2d(
              block_type,
              sample,
              timestep_embeds,
              residuals,
              encoder_last_hidden_state,
              num_layers: config.layers_per_block + 1,
              in_channels: in_channels,
              out_channels: out_channels,
              add_upsample: not last_block?,
              resnet_epsilon: config.norm_eps,
              resnet_num_groups: config.norm_num_groups,
              resnet_activation: config.act_fn,
              num_attention_heads: config.attention_head_dim,
              name: join(name, idx)
            )

          {sample, out_channels}
      end

    sample
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["act_fn"])
      |> Shared.convert_common()
      |> Shared.map_items("down_block_types", %{
        "DownBlock2D" => :down_block,
        "CrossAttnDownBlock2D" => :cross_attention_down_block
      })
      |> Shared.map_items("up_block_types", %{
        "UpBlock2D" => :up_block,
        "CrossAttnUpBlock2D" => :cross_attention_up_block
      })
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
