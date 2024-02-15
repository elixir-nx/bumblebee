defmodule Bumblebee.Diffusion.StableDiffusion.ControlNet do
  alias Bumblebee.Shared

  options = [
    sample_size: [
      default: 64,
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
      doc:
        "the number of attention heads for each attention layer. Optionally can be a list with one number per block"
    ],
    cross_attention_size: [
      default: 1024,
      doc: "the dimensionality of the cross attention features"
    ],
    use_linear_projection: [
      default: false,
      doc:
        "whether the input/output projection of the transformer block should be linear or convolutional"
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
    ],
    conditioning_embedding_out_channels: [
      default: [16, 32, 96, 256],
      doc: "the dimensionality of conditioning embedding"
    ]
  ]

  @moduledoc """
  ControlNet model with two spatial dimensions and conditional state.

  ## Architectures

    * `:base` - the ControlNet model

  ## Inputs

    * `"sample"` - `{batch_size, sample_size, sample_size, in_channels}`

      Sample input with two spatial dimensions.

    * `"timestep"` - `{}`

      The timestep used to parameterize model behaviour in a multi-step
      process, such as diffusion.

    * `"encoder_hidden_state"` - `{batch_size, sequence_length, hidden_size}`

      The conditional state (context) to use with cross-attention.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers
  alias Bumblebee.Diffusion

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(spec) do
    sample_shape = {1, spec.sample_size, spec.sample_size, spec.in_channels}
    timestep_shape = {}
    controlnet_conditioning_shape = {1, 512, 512, spec.in_channels}
    encoder_hidden_state_shape = {1, 1, spec.cross_attention_size}

    %{
      "sample" => Nx.template(sample_shape, :f32),
      "timestep" => Nx.template(timestep_shape, :u32),
      "controlnet_conditioning" => Nx.template(controlnet_conditioning_shape, :f32),
      "encoder_hidden_state" => Nx.template(encoder_hidden_state_shape, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs(spec)
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    sample_shape = {nil, spec.sample_size, spec.sample_size, spec.in_channels}
    controlnet_conditioning_shape = {nil, 512, 512, spec.in_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("sample", shape: sample_shape),
      Axon.input("timestep", shape: {}),
      Axon.input("controlnet_conditioning", shape: controlnet_conditioning_shape),
      Axon.input("encoder_hidden_state", shape: {nil, nil, spec.cross_attention_size})
    ])
  end

  defp core(inputs, spec) do
    sample = inputs["sample"]
    timestep = inputs["timestep"]
    controlnet_conditioning = inputs["controlnet_conditioning"]
    encoder_hidden_state = inputs["encoder_hidden_state"]

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
        name: "input_conv"
      )

    control_net_cond_embeddings =
      control_net_embeddings(controlnet_conditioning, spec, name: "controlnet_cond_embedding")

    sample =
      Axon.add(sample, control_net_cond_embeddings, name: "add_sample_control_net_embeddings")

    {sample, down_block_residuals} =
      down_blocks(sample, timestep_embedding, encoder_hidden_state, spec, name: "down_blocks")

    sample =
      mid_block(sample, timestep_embedding, encoder_hidden_state, spec, name: "mid_block")

    conditioning_scale = Axon.constant(1)

    down_block_residuals =
      control_net_down_blocks(down_block_residuals, spec, name: "controlnet_down_blocks")

    down_block_residuals =
      for residual <- Tuple.to_list(down_block_residuals) do
        Axon.multiply(residual, conditioning_scale, name: "conditioning_scale")
      end
      |> List.to_tuple()

    mid_block_residual =
      control_net_mid_block(sample, spec, name: "controlnet_mid_block")
      |> Axon.multiply(conditioning_scale)

    %{
      down_block_residuals: down_block_residuals,
      mid_block_residual: mid_block_residual
    }
  end

  defp control_net_down_blocks(down_block_residuals, spec, opts) do
    name = opts[:name]

    residuals =
      for {{residual, out_channels}, i} <- Enum.with_index(Tuple.to_list(down_block_residuals)) do
        Axon.conv(residual, out_channels,
          kernel_size: 1,
          padding: [{1, 1}, {1, 1}],
          name: name |> join(i) |> join("zero_conv"),
          kernel_initializer: :zeros
        )
      end

    List.to_tuple(residuals)
  end

  defp control_net_mid_block(input, spec, opts) do
    name = opts[:name]

    Axon.conv(input, List.last(spec.hidden_sizes),
      kernel_size: 1,
      padding: [{1, 1}, {1, 1}],
      name: name |> join("zero_conv"),
      kernel_initializer: :zeros
    )
  end

  defp control_net_embeddings(sample, spec, opts) do
    name = opts[:name]

    state =
      Axon.conv(sample, hd(spec.conditioning_embedding_out_channels),
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "input_conv"),
        activation: :silu
      )

    block_in_channels = Enum.drop(spec.conditioning_embedding_out_channels, -1)
    block_out_channels = Enum.drop(spec.conditioning_embedding_out_channels, 1)

    channels = Enum.zip(block_in_channels, block_out_channels)

    sample =
      for {{in_channels, out_channels}, i} <- Enum.with_index(channels),
          reduce: state do
        input ->
          input
          |> Axon.conv(in_channels,
            kernel_size: 3,
            padding: [{1, 1}, {1, 1}],
            name: name |> join(4 * i + 2) |> join("conv"),
            activation: :silu
          )
          |> Axon.conv(out_channels,
            kernel_size: 3,
            padding: [{1, 1}, {1, 1}],
            strides: 2,
            name: name |> join(4 * (i + 1)) |> join("conv"),
            activation: :silu
          )
      end

    Axon.conv(sample, hd(spec.hidden_sizes),
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "output_conv"),
      kernel_initializer: :zeros
    )
  end

  defp down_blocks(sample, timestep_embedding, encoder_hidden_state, spec, opts) do
    name = opts[:name]

    blocks =
      Enum.zip([spec.hidden_sizes, spec.down_block_types, num_attention_heads_per_block(spec)])

    in_channels = hd(spec.hidden_sizes)
    down_block_residuals = [{sample, in_channels}]

    state = {sample, down_block_residuals, in_channels}

    {sample, down_block_residuals, _} =
      for {{out_channels, block_type, num_attention_heads}, idx} <- Enum.with_index(blocks),
          reduce: state do
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
              num_attention_heads: num_attention_heads,
              use_linear_projection: spec.use_linear_projection,
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
      num_attention_heads: spec |> num_attention_heads_per_block() |> List.last(),
      use_linear_projection: spec.use_linear_projection,
      name: opts[:name]
    )
  end

  defp num_attention_heads_per_block(spec) when is_list(spec.num_attention_heads) do
    spec.num_attention_heads
  end

  defp num_attention_heads_per_block(spec) when is_integer(spec.num_attention_heads) do
    num_blocks = length(spec.down_block_types)
    List.duplicate(spec.num_attention_heads, num_blocks)
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
          num_attention_heads: {"attention_head_dim", one_of([number(), list(number())])},
          cross_attention_size: {"cross_attention_dim", number()},
          use_linear_projection: {"use_linear_projection", boolean()},
          activation: {"act_fn", activation()},
          group_norm_num_groups: {"norm_num_groups", number()},
          group_norm_epsilon: {"norm_eps", number()},
          conditioning_embedding_out_channels:
            {"conditioning_embedding_out_channels", list(number())}
        )

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      # controlnet_cond_embedding_mapping =
      %{
        "controlnet_cond_embedding.input_conv" => "control_model.input_hint_block.0",
        "controlnet_cond_embedding.output_conv" => "control_model.input_hint_block.14",
        "controlnet_cond_embedding.{l}.conv" => "control_model.input_hint_block.{l}",

        # controlnet_down_blocks_mapping = %{
        "controlnet_down_blocks.{m}.zero_conv" => "control_model.zero_convs.{m}.0",

        # controlnet_mid_block_mapping = %{
        "controlnet_mid_block.zero_conv" => "control_model.middle_block_out.0",

        # controlnet_mapping = %{
        "input_conv" => "control_model.input_blocks.0.0",

        # down_blocks_mapping = %{
        # down_blocks
        "down_blocks.0.transformers.0.norm" => "control_model.input_blocks.1.1.norm",
        "down_blocks.0.transformers.1.norm" => "control_model.input_blocks.2.1.norm",
        "down_blocks.1.transformers.0.norm" => "control_model.input_blocks.4.1.norm",
        "down_blocks.1.transformers.1.norm" => "control_model.input_blocks.5.1.norm",
        "down_blocks.2.transformers.0.norm" => "control_model.input_blocks.7.1.norm",
        "down_blocks.2.transformers.1.norm" => "control_model.input_blocks.8.1.norm",

        # self attention 0 0
        "down_blocks.0.transformers.0.blocks.0.self_attention_norm" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.norm1",
        "down_blocks.0.transformers.0.blocks.0.self_attention.key" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k",
        "down_blocks.0.transformers.0.blocks.0.self_attention.value" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v",
        "down_blocks.0.transformers.0.blocks.0.self_attention.query" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q",
        "down_blocks.0.transformers.0.blocks.0.self_attention.output" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0",

        # self attention 0 1
        "down_blocks.0.transformers.1.blocks.0.self_attention_norm" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.norm1",
        "down_blocks.0.transformers.1.blocks.0.self_attention.key" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k",
        "down_blocks.0.transformers.1.blocks.0.self_attention.value" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v",
        "down_blocks.0.transformers.1.blocks.0.self_attention.query" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q",
        "down_blocks.0.transformers.1.blocks.0.self_attention.output" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0",

        # self attention 1 0
        "down_blocks.1.transformers.0.blocks.0.self_attention_norm" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.norm1",
        "down_blocks.1.transformers.0.blocks.0.self_attention.key" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k",
        "down_blocks.1.transformers.0.blocks.0.self_attention.value" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v",
        "down_blocks.1.transformers.0.blocks.0.self_attention.query" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q",
        "down_blocks.1.transformers.0.blocks.0.self_attention.output" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0",

        # self attention 1 1
        "down_blocks.1.transformers.1.blocks.0.self_attention_norm" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.norm1",
        "down_blocks.1.transformers.1.blocks.0.self_attention.key" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k",
        "down_blocks.1.transformers.1.blocks.0.self_attention.value" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v",
        "down_blocks.1.transformers.1.blocks.0.self_attention.query" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q",
        "down_blocks.1.transformers.1.blocks.0.self_attention.output" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0",

        # self attention 2 0
        "down_blocks.2.transformers.0.blocks.0.self_attention_norm" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.norm1",
        "down_blocks.2.transformers.0.blocks.0.self_attention.key" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k",
        "down_blocks.2.transformers.0.blocks.0.self_attention.value" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v",
        "down_blocks.2.transformers.0.blocks.0.self_attention.query" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q",
        "down_blocks.2.transformers.0.blocks.0.self_attention.output" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0",

        # self attention 2 1
        "down_blocks.2.transformers.1.blocks.0.self_attention_norm" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.norm1",
        "down_blocks.2.transformers.1.blocks.0.self_attention.key" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k",
        "down_blocks.2.transformers.1.blocks.0.self_attention.value" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v",
        "down_blocks.2.transformers.1.blocks.0.self_attention.query" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q",
        "down_blocks.2.transformers.1.blocks.0.self_attention.output" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0",

        # cross attention 0 0
        "down_blocks.0.transformers.0.blocks.0.cross_attention_norm" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.norm2",
        "down_blocks.0.transformers.0.blocks.0.cross_attention.key" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k",
        "down_blocks.0.transformers.0.blocks.0.cross_attention.value" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v",
        "down_blocks.0.transformers.0.blocks.0.cross_attention.query" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q",
        "down_blocks.0.transformers.0.blocks.0.cross_attention.output" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0",

        # cross attention 0 1
        "down_blocks.0.transformers.1.blocks.0.cross_attention_norm" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.norm2",
        "down_blocks.0.transformers.1.blocks.0.cross_attention.key" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k",
        "down_blocks.0.transformers.1.blocks.0.cross_attention.value" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v",
        "down_blocks.0.transformers.1.blocks.0.cross_attention.query" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q",
        "down_blocks.0.transformers.1.blocks.0.cross_attention.output" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0",

        # cross attention 1 0
        "down_blocks.1.transformers.0.blocks.0.cross_attention_norm" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.norm2",
        "down_blocks.1.transformers.0.blocks.0.cross_attention.key" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k",
        "down_blocks.1.transformers.0.blocks.0.cross_attention.value" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v",
        "down_blocks.1.transformers.0.blocks.0.cross_attention.query" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q",
        "down_blocks.1.transformers.0.blocks.0.cross_attention.output" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0",

        # cross attention 1 1
        "down_blocks.1.transformers.1.blocks.0.cross_attention_norm" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.norm2",
        "down_blocks.1.transformers.1.blocks.0.cross_attention.key" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k",
        "down_blocks.1.transformers.1.blocks.0.cross_attention.value" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v",
        "down_blocks.1.transformers.1.blocks.0.cross_attention.query" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q",
        "down_blocks.1.transformers.1.blocks.0.cross_attention.output" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0",

        # cross attention 2 0
        "down_blocks.2.transformers.0.blocks.0.cross_attention_norm" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.norm2",
        "down_blocks.2.transformers.0.blocks.0.cross_attention.key" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k",
        "down_blocks.2.transformers.0.blocks.0.cross_attention.value" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v",
        "down_blocks.2.transformers.0.blocks.0.cross_attention.query" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q",
        "down_blocks.2.transformers.0.blocks.0.cross_attention.output" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0",

        # cross attention 2 1
        "down_blocks.2.transformers.1.blocks.0.cross_attention_norm" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.norm2",
        "down_blocks.2.transformers.1.blocks.0.cross_attention.key" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k",
        "down_blocks.2.transformers.1.blocks.0.cross_attention.value" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v",
        "down_blocks.2.transformers.1.blocks.0.cross_attention.query" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q",
        "down_blocks.2.transformers.1.blocks.0.cross_attention.output" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0",

        # ffn 0 0 
        "down_blocks.0.transformers.0.blocks.0.ffn.intermediate" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj",
        "down_blocks.0.transformers.0.blocks.0.ffn.output" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.ff.net.2",

        # ffn 0 1 
        "down_blocks.0.transformers.1.blocks.0.ffn.intermediate" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj",
        "down_blocks.0.transformers.1.blocks.0.ffn.output" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.ff.net.2",

        # ffn 1 0 
        "down_blocks.1.transformers.0.blocks.0.ffn.intermediate" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj",
        "down_blocks.1.transformers.0.blocks.0.ffn.output" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.ff.net.2",

        # ffn 1 1 
        "down_blocks.1.transformers.1.blocks.0.ffn.intermediate" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj",
        "down_blocks.1.transformers.1.blocks.0.ffn.output" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.ff.net.2",

        # ffn 2 0 
        "down_blocks.2.transformers.0.blocks.0.ffn.intermediate" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj",
        "down_blocks.2.transformers.0.blocks.0.ffn.output" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.ff.net.2",

        # ffn 2 1 
        "down_blocks.2.transformers.1.blocks.0.ffn.intermediate" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj",
        "down_blocks.2.transformers.1.blocks.0.ffn.output" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.ff.net.2",

        # residuals 0 0
        "down_blocks.0.residual_blocks.0.norm_1" => "control_model.input_blocks.1.0.in_layers.0",
        "down_blocks.0.residual_blocks.0.conv_1" => "control_model.input_blocks.1.0.in_layers.2",
        "down_blocks.0.residual_blocks.0.timestep_projection" =>
          "control_model.input_blocks.1.0.emb_layers.1",
        "down_blocks.0.residual_blocks.0.norm_2" => "control_model.input_blocks.1.0.out_layers.0",
        "down_blocks.0.residual_blocks.0.conv_2" => "control_model.input_blocks.1.0.out_layers.3",

        # residuals 0 1
        "down_blocks.0.residual_blocks.1.norm_1" => "control_model.input_blocks.2.0.in_layers.0",
        "down_blocks.0.residual_blocks.1.conv_1" => "control_model.input_blocks.2.0.in_layers.2",
        "down_blocks.0.residual_blocks.1.timestep_projection" =>
          "control_model.input_blocks.2.0.emb_layers.1",
        "down_blocks.0.residual_blocks.1.norm_2" => "control_model.input_blocks.2.0.out_layers.0",
        "down_blocks.0.residual_blocks.1.conv_2" => "control_model.input_blocks.2.0.out_layers.3",

        # residuals 1 0
        "down_blocks.1.residual_blocks.0.norm_1" => "control_model.input_blocks.4.0.in_layers.0",
        "down_blocks.1.residual_blocks.0.conv_1" => "control_model.input_blocks.4.0.in_layers.2",
        "down_blocks.1.residual_blocks.0.timestep_projection" =>
          "control_model.input_blocks.4.0.emb_layers.1",
        "down_blocks.1.residual_blocks.0.norm_2" => "control_model.input_blocks.4.0.out_layers.0",
        "down_blocks.1.residual_blocks.0.conv_2" => "control_model.input_blocks.4.0.out_layers.3",

        # residuals 1 1
        "down_blocks.1.residual_blocks.1.norm_1" => "control_model.input_blocks.5.0.in_layers.0",
        "down_blocks.1.residual_blocks.1.conv_1" => "control_model.input_blocks.5.0.in_layers.2",
        "down_blocks.1.residual_blocks.1.timestep_projection" =>
          "control_model.input_blocks.5.0.emb_layers.1",
        "down_blocks.1.residual_blocks.1.norm_2" => "control_model.input_blocks.5.0.out_layers.0",
        "down_blocks.1.residual_blocks.1.conv_2" => "control_model.input_blocks.5.0.out_layers.3",

        # residuals 2 0
        "down_blocks.2.residual_blocks.0.norm_1" => "control_model.input_blocks.7.0.in_layers.0",
        "down_blocks.2.residual_blocks.0.conv_1" => "control_model.input_blocks.7.0.in_layers.2",
        "down_blocks.2.residual_blocks.0.timestep_projection" =>
          "control_model.input_blocks.7.0.emb_layers.1",
        "down_blocks.2.residual_blocks.0.norm_2" => "control_model.input_blocks.7.0.out_layers.0",
        "down_blocks.2.residual_blocks.0.conv_2" => "control_model.input_blocks.7.0.out_layers.3",

        # residuals 2 1
        "down_blocks.2.residual_blocks.1.norm_1" => "control_model.input_blocks.8.0.in_layers.0",
        "down_blocks.2.residual_blocks.1.conv_1" => "control_model.input_blocks.8.0.in_layers.2",
        "down_blocks.2.residual_blocks.1.timestep_projection" =>
          "control_model.input_blocks.8.0.emb_layers.1",
        "down_blocks.2.residual_blocks.1.norm_2" => "control_model.input_blocks.8.0.out_layers.0",
        "down_blocks.2.residual_blocks.1.conv_2" => "control_model.input_blocks.8.0.out_layers.3",

        # residuals 3 0
        "down_blocks.3.residual_blocks.0.norm_1" => "control_model.input_blocks.10.0.in_layers.0",
        "down_blocks.3.residual_blocks.0.conv_1" => "control_model.input_blocks.10.0.in_layers.2",
        "down_blocks.3.residual_blocks.0.timestep_projection" =>
          "control_model.input_blocks.10.0.emb_layers.1",
        "down_blocks.3.residual_blocks.0.norm_2" =>
          "control_model.input_blocks.10.0.out_layers.0",
        "down_blocks.3.residual_blocks.0.conv_2" =>
          "control_model.input_blocks.10.0.out_layers.3",

        # residuals 3 1
        "down_blocks.3.residual_blocks.1.norm_1" => "control_model.input_blocks.11.0.in_layers.0",
        "down_blocks.3.residual_blocks.1.conv_1" => "control_model.input_blocks.11.0.in_layers.2",
        "down_blocks.3.residual_blocks.1.timestep_projection" =>
          "control_model.input_blocks.11.0.emb_layers.1",
        "down_blocks.3.residual_blocks.1.norm_2" =>
          "control_model.input_blocks.11.0.out_layers.0",
        "down_blocks.3.residual_blocks.1.conv_2" =>
          "control_model.input_blocks.11.0.out_layers.3",

        # projection 0 0
        "down_blocks.0.transformers.0.input_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.1.1.proj_in", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.1.1.proj_in", "bias"}], fn [value] -> value end}
        },
        "down_blocks.0.transformers.0.output_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.1.1.proj_out", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.1.1.proj_out", "bias"}], fn [value] -> value end}
        },

        # projection 0 1
        "down_blocks.0.transformers.1.input_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.2.1.proj_in", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.2.1.proj_in", "bias"}], fn [value] -> value end}
        },
        "down_blocks.0.transformers.1.output_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.2.1.proj_out", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.2.1.proj_out", "bias"}], fn [value] -> value end}
        },

        # projection 1 0
        "down_blocks.1.transformers.0.input_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.4.1.proj_in", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.4.1.proj_in", "bias"}], fn [value] -> value end}
        },
        "down_blocks.1.transformers.0.output_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.4.1.proj_out", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.4.1.proj_out", "bias"}], fn [value] -> value end}
        },

        # projection 1 1
        "down_blocks.1.transformers.1.input_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.5.1.proj_in", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.5.1.proj_in", "bias"}], fn [value] -> value end}
        },
        "down_blocks.1.transformers.1.output_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.5.1.proj_out", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.5.1.proj_out", "bias"}], fn [value] -> value end}
        },

        # projection 2 0
        "down_blocks.2.transformers.0.input_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.7.1.proj_in", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.7.1.proj_in", "bias"}], fn [value] -> value end}
        },
        "down_blocks.2.transformers.0.output_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.7.1.proj_out", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.7.1.proj_out", "bias"}], fn [value] -> value end}
        },

        # projection 2 1
        "down_blocks.2.transformers.1.input_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.8.1.proj_in", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.8.1.proj_in", "bias"}], fn [value] -> value end}
        },
        "down_blocks.2.transformers.1.output_projection" => %{
          "kernel" =>
            {[{"control_model.input_blocks.8.1.proj_out", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" =>
            {[{"control_model.input_blocks.8.1.proj_out", "bias"}], fn [value] -> value end}
        },

        # shortcut
        "down_blocks.1.residual_blocks.0.shortcut.projection" =>
          "control_model.input_blocks.4.0.skip_connection",
        "down_blocks.2.residual_blocks.0.shortcut.projection" =>
          "control_model.input_blocks.7.0.skip_connection",

        # downsamples
        "down_blocks.0.downsamples.0.conv" => "control_model.input_blocks.3.0.op",
        "down_blocks.1.downsamples.0.conv" => "control_model.input_blocks.6.0.op",
        "down_blocks.2.downsamples.0.conv" => "control_model.input_blocks.9.0.op",

        # out 0 0
        "down_blocks.0.transformers.0.blocks.0.output_norm" =>
          "control_model.input_blocks.1.1.transformer_blocks.0.norm3",

        # out 0 1
        "down_blocks.0.transformers.1.blocks.0.output_norm" =>
          "control_model.input_blocks.2.1.transformer_blocks.0.norm3",

        # out 1 0
        "down_blocks.1.transformers.0.blocks.0.output_norm" =>
          "control_model.input_blocks.4.1.transformer_blocks.0.norm3",

        # out 1 1
        "down_blocks.1.transformers.1.blocks.0.output_norm" =>
          "control_model.input_blocks.5.1.transformer_blocks.0.norm3",

        # out 2 0
        "down_blocks.2.transformers.0.blocks.0.output_norm" =>
          "control_model.input_blocks.7.1.transformer_blocks.0.norm3",

        # out 2 1
        "down_blocks.2.transformers.1.blocks.0.output_norm" =>
          "control_model.input_blocks.8.1.transformer_blocks.0.norm3",

        # mid_block_mapping = %{
        # mid_block
        "mid_block.transformers.0.norm" => "control_model.middle_block.1.norm",
        # self attention
        "mid_block.transformers.0.blocks.0.self_attention_norm" =>
          "control_model.middle_block.1.transformer_blocks.0.norm1",
        "mid_block.transformers.0.blocks.0.self_attention.key" =>
          "control_model.middle_block.1.transformer_blocks.0.attn1.to_k",
        "mid_block.transformers.0.blocks.0.self_attention.value" =>
          "control_model.middle_block.1.transformer_blocks.0.attn1.to_v",
        "mid_block.transformers.0.blocks.0.self_attention.query" =>
          "control_model.middle_block.1.transformer_blocks.0.attn1.to_q",
        "mid_block.transformers.0.blocks.0.self_attention.output" =>
          "control_model.middle_block.1.transformer_blocks.0.attn1.to_out.0",

        # cross attention
        "mid_block.transformers.0.blocks.0.cross_attention_norm" =>
          "control_model.middle_block.1.transformer_blocks.0.norm2",
        "mid_block.transformers.0.blocks.0.cross_attention.key" =>
          "control_model.middle_block.1.transformer_blocks.0.attn2.to_k",
        "mid_block.transformers.0.blocks.0.cross_attention.value" =>
          "control_model.middle_block.1.transformer_blocks.0.attn2.to_v",
        "mid_block.transformers.0.blocks.0.cross_attention.query" =>
          "control_model.middle_block.1.transformer_blocks.0.attn2.to_q",
        "mid_block.transformers.0.blocks.0.cross_attention.output" =>
          "control_model.middle_block.1.transformer_blocks.0.attn2.to_out.0",

        # ffn
        "mid_block.transformers.0.blocks.0.ffn.intermediate" =>
          "control_model.middle_block.1.transformer_blocks.0.ff.net.0.proj",
        "mid_block.transformers.0.blocks.0.ffn.output" =>
          "control_model.middle_block.1.transformer_blocks.0.ff.net.2",

        # residuals 0
        "mid_block.residual_blocks.0.norm_1" => "control_model.middle_block.0.in_layers.0",
        "mid_block.residual_blocks.0.conv_1" => "control_model.middle_block.0.in_layers.2",
        "mid_block.residual_blocks.0.timestep_projection" =>
          "control_model.middle_block.0.emb_layers.1",
        "mid_block.residual_blocks.0.norm_2" => "control_model.middle_block.0.out_layers.0",
        "mid_block.residual_blocks.0.conv_2" => "control_model.middle_block.0.out_layers.3",
        # residuals 1
        "mid_block.residual_blocks.1.norm_1" => "control_model.middle_block.2.in_layers.0",
        "mid_block.residual_blocks.1.conv_1" => "control_model.middle_block.2.in_layers.2",
        "mid_block.residual_blocks.1.timestep_projection" =>
          "control_model.middle_block.2.emb_layers.1",
        "mid_block.residual_blocks.1.norm_2" => "control_model.middle_block.2.out_layers.0",
        "mid_block.residual_blocks.1.conv_2" => "control_model.middle_block.2.out_layers.3",

        # projection
        "mid_block.transformers.0.input_projection" => %{
          "kernel" =>
            {[{"control_model.middle_block.1.proj_in", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" => {[{"control_model.middle_block.1.proj_in", "bias"}], fn [value] -> value end}
        },
        "mid_block.transformers.0.output_projection" => %{
          "kernel" =>
            {[{"control_model.middle_block.1.proj_out", "weight"}],
             fn [value] -> value |> Nx.new_axis(0) |> Nx.new_axis(0) end},
          "bias" => {[{"control_model.middle_block.1.proj_out", "bias"}], fn [value] -> value end}
        },

        # out
        "mid_block.transformers.0.blocks.0.output_norm" =>
          "control_model.middle_block.1.transformer_blocks.0.norm3",

        # others
        "time_embedding.intermediate" => "control_model.time_embed.0",
        "time_embedding.output" => "control_model.time_embed.2"
      }
    end
  end
end
