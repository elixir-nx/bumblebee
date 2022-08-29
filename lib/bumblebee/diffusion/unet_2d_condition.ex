defmodule Bumblebee.Diffusion.UNet2DCondition do
  @moduledoc """
  UNet 2D Conditional model.
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Diffusion

  defstruct architecture: :base,
            in_channels: 4,
            out_channels: 4,
            height: 512,
            width: 512,
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0,
            down_block_types: [
              "CrossAttnDownBlock2D",
              "CrossAttnDownBlock2D",
              "CrossAttnDownBlock2D",
              "DownBlock2D"
            ],
            up_block_types: [
              "UpBlock2D",
              "CrossAttnUpBlock2D",
              "CrossAttnUpBlock2D",
              "CrossAttnUpBlock2D"
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
    sample_shape = {1, config.in_channels, config.height, config.width}
    timestep_shape = {}
    encoder_hidden_states_shape = {1, 1, config.cross_attention_dim}

    %{
      "sample" => Nx.template(sample_shape, :f32),
      "timestep" => Nx.template(timestep_shape, :s64),
      # TODO: Rename encoder_hidden_state
      "encoder_hidden_states" => Nx.template(encoder_hidden_states_shape, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs = inputs(config)
    unet_2d_condition(inputs, config, name: "unet")
  end

  defp inputs(config) do
    sample_shape = {nil, config.in_channels, config.height, config.width}

    %{
      "sample" => Axon.input("sample", shape: sample_shape),
      "timestep" => Axon.input("timestep", shape: {}),
      "encoder_hidden_states" => Axon.input("encoder_hidden_states")
    }
  end

  defp unet_2d_condition(inputs, config, opts) do
    name = opts[:name]

    sample = inputs["sample"]
    timestep = inputs["timestep"]
    encoder_hidden_states = inputs["encoder_hidden_states"]

    sample =
      if config.center_input_sample do
        Axon.nx(sample, fn sample -> 2 * sample - 1.0 end, op_name: :center)
      else
        sample
      end

    timesteps =
      Axon.layer(
        fn sample, timestep, _opts ->
          Nx.broadcast(timestep, Nx.axis_size(sample, 0))
        end,
        [sample, timestep],
        op_name: :broadcast
      )

    t_emb =
      Diffusion.Layers.timesteps(timesteps,
        num_channels: Enum.at(config.block_out_channels, 0),
        flip_sin_to_cos: config.flip_sin_to_cos,
        downscale_freq_shift: config.freq_shift,
        op_name: :timesteps
      )

    emb =
      Diffusion.Layers.timestep_embedding(t_emb, Enum.at(config.block_out_channels, 0) * 4,
        name: "time_embedding"
      )

    sample =
      Axon.conv(sample, Enum.at(config.block_out_channels, 0),
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv_in")
      )

    {sample, down_block_res_samples} =
      down_blocks(sample, emb, encoder_hidden_states, config, name: join(name, "down_blocks"))

    sample
    |> mid_block(emb, encoder_hidden_states, config, name: join(name, "mid_block"))
    |> up_blocks(emb, down_block_res_samples, encoder_hidden_states, config,
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

  defp down_blocks(sample, timestep_embedding, encoder_hidden_states, config, opts) do
    name = opts[:name]
    blocks = Enum.zip(config.block_out_channels, config.down_block_types)

    down_block_res_samples = [sample]
    in_channels = Enum.at(config.block_out_channels, 0)
    time_embed_dim = in_channels * 4

    state = {sample, down_block_res_samples, in_channels}

    {sample, down_block_res_samples, _} =
      for {{out_channels, block_type}, idx} <- Enum.with_index(blocks), reduce: state do
        {sample, down_block_res_samples, in_channels} ->
          block_opts = [
            num_layers: config.layers_per_block,
            in_channels: in_channels,
            out_channels: out_channels,
            temb_channels: time_embed_dim,
            add_downsample: idx != Enum.count(config.block_out_channels) - 1,
            resnet_eps: config.norm_eps,
            resnet_act_fn: config.act_fn,
            cross_attention_dim: config.cross_attention_dim,
            attn_num_head_channels: config.attention_head_dim,
            downsample_padding: config.downsample_padding,
            name: join(name, "#{idx}")
          ]

          {sample, res_samples} =
            Diffusion.Layers.apply_unet_block(block_type, [
              sample,
              timestep_embedding,
              encoder_hidden_states,
              block_opts
            ])

          {sample, down_block_res_samples ++ Tuple.to_list(res_samples), out_channels}
      end

    {sample, List.to_tuple(down_block_res_samples)}
  end

  defp mid_block(hidden_state, timesteps_embedding, encoder_hidden_states, config, opts) do
    in_channels = Enum.at(config.block_out_channels, 0)
    time_embed_dim = in_channels * 4

    Diffusion.Layers.mid_block_2d_cross_attn(
      hidden_state,
      timesteps_embedding,
      encoder_hidden_states,
      in_channels: Enum.at(config.block_out_channels, Enum.count(config.block_out_channels) - 1),
      temb_channels: time_embed_dim,
      resnet_eps: config.norm_eps,
      resnet_act_fn: config.act_fn,
      output_scale_factor: config.mid_block_scale_factor,
      resnet_time_scale_shift: :default,
      cross_attention_dim: config.cross_attention_dim,
      attn_num_head_channels: config.attention_head_dim,
      resnet_groups: config.norm_num_groups,
      name: opts[:name]
    )
  end

  defp up_blocks(
         sample,
         timestep_embedding,
         down_block_res_samples,
         encoder_hidden_states,
         config,
         opts
       ) do
    name = opts[:name]
    in_channels = Enum.at(config.block_out_channels, 0)
    time_embed_dim = in_channels * 4

    down_block_res_samples =
      down_block_res_samples
      |> Tuple.to_list()
      |> Enum.reverse()
      |> Enum.chunk_every(config.layers_per_block + 1)

    reversed_block_out_channels = Enum.reverse(config.block_out_channels)
    in_channels = Enum.at(reversed_block_out_channels, 0)

    blocks_and_chunks =
      [reversed_block_out_channels, config.up_block_types, down_block_res_samples]
      |> Enum.zip()
      |> Enum.with_index()

    {sample, _} =
      for {{out_channels, block_type, res_samples}, idx} <- blocks_and_chunks,
          reduce: {sample, in_channels} do
        {sample, in_channels} ->
          block_opts = [
            num_layers: config.layers_per_block + 1,
            in_channels: in_channels,
            out_channels: out_channels,
            prev_output_channel: in_channels,
            temb_channels: time_embed_dim,
            add_upsample: idx != Enum.count(config.block_out_channels) - 1,
            resnet_eps: config.norm_eps,
            resnet_act_fn: config.act_fn,
            cross_attention_dim: config.cross_attention_dim,
            attn_num_head_channels: config.attention_head_dim,
            name: join(name, "#{idx}")
          ]

          {Diffusion.Layers.apply_unet_block(block_type, [
             sample,
             timestep_embedding,
             res_samples,
             encoder_hidden_states,
             block_opts
           ]), out_channels}
      end

    sample
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
