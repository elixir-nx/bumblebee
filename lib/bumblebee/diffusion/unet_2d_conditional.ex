defmodule Bumblebee.Diffusion.UNet2DConditional do
  alias Bumblebee.Shared

  options = [
    sample_size: [
      default: 32,
      doc: "the size of the input spatial dimensions"
    ],
    in_channels: [
      default: 4,
      doc: "the number of channels in the input"
    ],
    out_channels: [
      default: 4,
      doc: "the number of channels in the output"
    ],
    center_input_sample: [
      default: false,
      doc: "whether to center the input sample"
    ],
    embedding_flip_sin_to_cos: [
      default: true,
      doc: "whether to flip the sin to cos in the sinusoidal timestep embedding"
    ],
    embedding_frequency_correction_term: [
      default: 0,
      doc: ~S"""
      controls the frequency formula in the timestep sinusoidal embedding. The frequency is computed
      as $\\omega_i = \\frac{1}{10000^{\\frac{i}{n - s}}}$, for $i \\in \\{0, ..., n-1\\}$, where $n$
      is half of the embedding size and $s$ is the shift. Historically, certain implementations of
      sinusoidal embedding used $s=0$, while others used $s=1$
      """
    ],
    hidden_sizes: [
      default: [320, 640, 1280, 1280],
      doc: "the dimensionality of hidden layers in each upsample/downsample block"
    ],
    depth: [
      default: 2,
      doc: "the number of residual blocks in each upsample/downsample block"
    ],
    down_block_types: [
      default: [
        :cross_attention_down_block,
        :cross_attention_down_block,
        :cross_attention_down_block,
        :down_block
      ],
      doc:
        "a list of downsample block types. The supported blocks are: `:down_block`, `:cross_attention_down_block`"
    ],
    up_block_types: [
      default: [
        :up_block,
        :cross_attention_up_block,
        :cross_attention_up_block,
        :cross_attention_up_block
      ],
      doc:
        "a list of upsample block types. The supported blocks are: `:up_block`, `:cross_attention_up_block`"
    ],
    downsample_padding: [
      default: [{1, 1}, {1, 1}],
      doc: "the padding to use in the downsample convolution"
    ],
    mid_block_scale_factor: [
      default: 1,
      doc: "the scale factor to use for the mid block"
    ],
    num_attention_heads: [
      default: 8,
      doc: "the number of attention heads for each attention layer"
    ],
    cross_attention_size: [
      default: 1280,
      doc: "the dimensionality of the cross attention features"
    ],
    activation: [
      default: :silu,
      doc: "the activation function"
    ],
    group_norm_num_groups: [
      default: 32,
      doc: "the number of groups used by the group normalization layers"
    ],
    group_norm_epsilon: [
      default: 1.0e-5,
      doc: "the epsilon used by the group normalization layers"
    ]
  ]

  @moduledoc """
  U-Net model with two spatial dimensions and conditional state.

  ## Architectures

    * `:base` - the U-Net model

  ## Inputs

    * `"sample"` - `{batch_size, sample_size, sample_size, in_channels}`

      Sample input with two spatial dimensions.

    * `"timestep"` - `{}`

      The timestep used to parameterize model behaviour in a multi-step
      process, such as diffusion.

    * `"encoder_hidden_state"` - `{batch_size, seq_length, hidden_size}`

      The conditional state (context) to use with cross-attention.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers
  alias Bumblebee.Diffusion

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts \\ []) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(spec) do
    sample_shape = {1, spec.sample_size, spec.sample_size, spec.in_channels}
    timestep_shape = {}
    encoder_hidden_state_shape = {1, 1, spec.cross_attention_size}

    %{
      "sample" => Nx.template(sample_shape, :f32),
      "timestep" => Nx.template(timestep_shape, :s64),
      "encoder_hidden_state" => Nx.template(encoder_hidden_state_shape, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)
    sample = unet_2d_conditional(inputs, spec)
    Layers.output(%{sample: sample})
  end

  defp inputs(spec) do
    sample_shape = {nil, spec.sample_size, spec.sample_size, spec.in_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("sample", shape: sample_shape),
      Axon.input("timestep", shape: {}),
      Axon.input("encoder_hidden_state", shape: {nil, nil, spec.cross_attention_size})
    ])
  end

  defp unet_2d_conditional(inputs, spec, opts \\ []) do
    name = opts[:name]

    sample = inputs["sample"]
    timestep = inputs["timestep"]
    encoder_hidden_state = inputs["encoder_hidden_state"]

    sample =
      if spec.center_input_sample do
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

    timestep_embedding =
      timestep
      |> Diffusion.Layers.timestep_sinusoidal_embedding(hd(spec.hidden_sizes),
        flip_sin_to_cos: spec.embedding_flip_sin_to_cos,
        frequency_correction_term: spec.embedding_frequency_correction_term
      )
      |> Diffusion.Layers.UNet.timestep_embedding_mlp(hd(spec.hidden_sizes) * 4,
        name: "time_embedding"
      )

    sample =
      Axon.conv(sample, hd(spec.hidden_sizes),
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "conv_in")
      )

    {sample, down_block_residuals} =
      down_blocks(sample, timestep_embedding, encoder_hidden_state, spec,
        name: join(name, "down_blocks")
      )

    sample
    |> mid_block(timestep_embedding, encoder_hidden_state, spec, name: join(name, "mid_block"))
    |> up_blocks(timestep_embedding, down_block_residuals, encoder_hidden_state, spec,
      name: join(name, "up_blocks")
    )
    |> Axon.group_norm(spec.group_norm_num_groups,
      epsilon: spec.group_norm_epsilon,
      name: join(name, "conv_norm_out")
    )
    |> Axon.activation(:silu, name: join(name, "conv_act"))
    |> Axon.conv(spec.out_channels,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "conv_out")
    )
  end

  defp down_blocks(sample, timestep_embedding, encoder_hidden_state, spec, opts) do
    name = opts[:name]
    blocks = Enum.zip(spec.hidden_sizes, spec.down_block_types)

    in_channels = hd(spec.hidden_sizes)
    down_block_residuals = [{sample, in_channels}]

    state = {sample, down_block_residuals, in_channels}

    {sample, down_block_residuals, _} =
      for {{out_channels, block_type}, idx} <- Enum.with_index(blocks), reduce: state do
        {sample, down_block_residuals, in_channels} ->
          last_block? = idx == length(spec.hidden_sizes) - 1

          {sample, residuals} =
            Diffusion.Layers.UNet.down_block_2d(
              block_type,
              sample,
              timestep_embedding,
              encoder_hidden_state,
              depth: spec.depth,
              in_channels: in_channels,
              out_channels: out_channels,
              add_downsample: not last_block?,
              downsample_padding: spec.downsample_padding,
              activation: spec.activation,
              norm_epsilon: spec.group_norm_epsilon,
              norm_num_groups: spec.group_norm_num_groups,
              num_attention_heads: spec.num_attention_heads,
              name: join(name, idx)
            )

          {sample, down_block_residuals ++ Tuple.to_list(residuals), out_channels}
      end

    {sample, List.to_tuple(down_block_residuals)}
  end

  defp mid_block(hidden_state, timesteps_embedding, encoder_hidden_state, spec, opts) do
    Diffusion.Layers.UNet.mid_cross_attention_block_2d(
      hidden_state,
      timesteps_embedding,
      encoder_hidden_state,
      channels: List.last(spec.hidden_sizes),
      activation: spec.activation,
      norm_epsilon: spec.group_norm_epsilon,
      norm_num_groups: spec.group_norm_num_groups,
      output_scale_factor: spec.mid_block_scale_factor,
      num_attention_heads: spec.num_attention_heads,
      name: opts[:name]
    )
  end

  defp up_blocks(
         sample,
         timestep_embedding,
         down_block_residuals,
         encoder_hidden_state,
         spec,
         opts
       ) do
    name = opts[:name]

    down_block_residuals =
      down_block_residuals
      |> Tuple.to_list()
      |> Enum.reverse()
      |> Enum.chunk_every(spec.depth + 1)

    reversed_hidden_sizes = Enum.reverse(spec.hidden_sizes)
    in_channels = hd(reversed_hidden_sizes)

    blocks_and_chunks =
      [reversed_hidden_sizes, spec.up_block_types, down_block_residuals]
      |> Enum.zip()
      |> Enum.with_index()

    {sample, _} =
      for {{out_channels, block_type, residuals}, idx} <- blocks_and_chunks,
          reduce: {sample, in_channels} do
        {sample, in_channels} ->
          last_block? = idx == length(spec.hidden_sizes) - 1

          sample =
            Diffusion.Layers.UNet.up_block_2d(
              block_type,
              sample,
              timestep_embedding,
              residuals,
              encoder_hidden_state,
              depth: spec.depth + 1,
              in_channels: in_channels,
              out_channels: out_channels,
              add_upsample: not last_block?,
              norm_epsilon: spec.group_norm_epsilon,
              norm_num_groups: spec.group_norm_num_groups,
              activation: spec.activation,
              num_attention_heads: spec.num_attention_heads,
              name: join(name, idx)
            )

          {sample, out_channels}
      end

    sample
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          in_channels: {"in_channels", number()},
          out_channels: {"out_channels", number()},
          sample_size: {"sample_size", number()},
          center_input_sample: {"center_input_sample", boolean()},
          embedding_flip_sin_to_cos: {"flip_sin_to_cos", boolean()},
          embedding_frequency_correction_term: {"freq_shift", number()},
          hidden_sizes: {"block_out_channels", list(number())},
          depth: {"layers_per_block", number()},
          down_block_types: {
            "down_block_types",
            list(
              mapping(%{
                "DownBlock2D" => :down_block,
                "CrossAttnDownBlock2D" => :cross_attention_down_block
              })
            )
          },
          up_block_types: {
            "up_block_types",
            list(
              mapping(%{
                "UpBlock2D" => :up_block,
                "CrossAttnUpBlock2D" => :cross_attention_up_block
              })
            )
          },
          downsample_padding: {"downsample_padding", padding(2)},
          mid_block_scale_factor: {"mid_block_scale_factor", number()},
          num_attention_heads: {"attention_head_dim", number()},
          cross_attention_size: {"cross_attention_dim", number()},
          activation: {"act_fn", atom()},
          group_norm_num_groups: {"norm_num_groups", number()},
          group_norm_epsilon: {"norm_eps", number()}
        )

      @for.config(spec, opts)
    end
  end
end
