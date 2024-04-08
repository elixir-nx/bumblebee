defmodule Bumblebee.Diffusion.ControlNet do
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
    conditioning_embedding_hidden_sizes: [
      default: [16, 32, 96, 256],
      doc: "the dimensionality of hidden layers in the conditioning input embedding"
    ]
  ]

  @moduledoc """
  ControlNet model with two spatial dimensions and conditioning state.

  ## Architectures

    * `:base` - the ControlNet model

  ## Inputs

    * `"sample"` - `{batch_size, sample_size, sample_size, in_channels}`

      Sample input with two spatial dimensions.

    * `"timestep"` - `{}`

      The timestep used to parameterize model behaviour in a multi-step
      process, such as diffusion.

    * `"encoder_hidden_state"` - `{batch_size, sequence_length, hidden_size}`

      The conditioning state (context) to use with cross-attention.

    * `"conditioning"` - `{batch_size, conditioning_size, conditioning_size, 3}`

      The conditioning spatial input.

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

    conditioning_size =
      spec.sample_size * 2 ** (length(spec.conditioning_embedding_hidden_sizes) - 1)

    conditioning_shape = {1, conditioning_size, conditioning_size, 3}
    encoder_hidden_state_shape = {1, 1, spec.cross_attention_size}

    %{
      "sample" => Nx.template(sample_shape, :f32),
      "timestep" => Nx.template(timestep_shape, :u32),
      "conditioning" => Nx.template(conditioning_shape, :f32),
      "conditioning_scale" => Nx.template({}, :f32),
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

    conditioning_size =
      spec.sample_size * 2 ** (length(spec.conditioning_embedding_hidden_sizes) - 1)

    conditioning_shape = {nil, conditioning_size, conditioning_size, 3}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("sample", shape: sample_shape),
      Axon.input("timestep", shape: {}),
      Axon.input("conditioning", shape: conditioning_shape),
      Axon.input("conditioning_scale", optional: true),
      Axon.input("encoder_hidden_state", shape: {nil, nil, spec.cross_attention_size})
    ])
  end

  defp core(inputs, spec) do
    sample = inputs["sample"]
    timestep = inputs["timestep"]
    conditioning = inputs["conditioning"]

    conditioning_scale =
      Bumblebee.Layers.default inputs["conditioning_scale"] do
        Axon.constant(1)
      end

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

    controlnet_conditioning_embeddings =
      controlnet_embedding(conditioning, spec, name: "controlnet_conditioning_embedding")

    sample = Axon.add(sample, controlnet_conditioning_embeddings)

    {sample, down_blocks_residuals} =
      down_blocks(sample, timestep_embedding, encoder_hidden_state, spec, name: "down_blocks")

    sample =
      mid_block(sample, timestep_embedding, encoder_hidden_state, spec, name: "mid_block")

    down_blocks_residuals =
      controlnet_down_blocks(down_blocks_residuals, name: "controlnet_down_blocks")

    down_blocks_residuals =
      for residual <- Tuple.to_list(down_blocks_residuals) do
        Axon.multiply(residual, conditioning_scale, name: "down_conditioning_scale")
      end
      |> List.to_tuple()

    mid_block_residual =
      sample
      |> controlnet_mid_block(spec, name: "controlnet_mid_block")
      |> Axon.multiply(conditioning_scale)

    %{
      down_blocks_residuals: Axon.container(down_blocks_residuals),
      mid_block_residual: mid_block_residual
    }
  end

  defp controlnet_down_blocks(down_block_residuals, opts) do
    name = opts[:name]

    residuals =
      for {{residual, out_channels}, i} <- Enum.with_index(Tuple.to_list(down_block_residuals)) do
        Axon.conv(residual, out_channels,
          kernel_size: 1,
          name: name |> join(i) |> join("zero_conv"),
          kernel_initializer: :zeros
        )
      end

    List.to_tuple(residuals)
  end

  defp controlnet_mid_block(input, spec, opts) do
    name = opts[:name]

    Axon.conv(input, List.last(spec.hidden_sizes),
      kernel_size: 1,
      name: name |> join("zero_conv"),
      kernel_initializer: :zeros
    )
  end

  defp controlnet_embedding(sample, spec, opts) do
    name = opts[:name]

    state =
      Axon.conv(sample, hd(spec.conditioning_embedding_hidden_sizes),
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        name: join(name, "input_conv"),
        activation: :silu
      )

    block_in_channels = Enum.drop(spec.conditioning_embedding_hidden_sizes, -1)
    block_out_channels = Enum.drop(spec.conditioning_embedding_hidden_sizes, 1)

    channels = Enum.zip(block_in_channels, block_out_channels)

    sample =
      for {{in_channels, out_channels}, i} <- Enum.with_index(channels),
          reduce: state do
        input ->
          input
          |> Axon.conv(in_channels,
            kernel_size: 3,
            padding: [{1, 1}, {1, 1}],
            name: name |> join("inner_convs") |> join(2 * i),
            activation: :silu
          )
          |> Axon.conv(out_channels,
            kernel_size: 3,
            padding: [{1, 1}, {1, 1}],
            strides: 2,
            name: name |> join("inner_convs") |> join(2 * i + 1),
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
          conditioning_embedding_hidden_sizes:
            {"conditioning_embedding_out_channels", list(number())}
        )

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.HuggingFace.Transformers

    def params_mapping(_spec) do
      block_mapping = %{
        "transformers.{m}.norm" => "attentions.{m}.norm",
        "transformers.{m}.input_projection" => "attentions.{m}.proj_in",
        "transformers.{m}.output_projection" => "attentions.{m}.proj_out",
        "transformers.{m}.blocks.{l}.self_attention.query" =>
          "attentions.{m}.transformer_blocks.{l}.attn1.to_q",
        "transformers.{m}.blocks.{l}.self_attention.key" =>
          "attentions.{m}.transformer_blocks.{l}.attn1.to_k",
        "transformers.{m}.blocks.{l}.self_attention.value" =>
          "attentions.{m}.transformer_blocks.{l}.attn1.to_v",
        "transformers.{m}.blocks.{l}.self_attention.output" =>
          "attentions.{m}.transformer_blocks.{l}.attn1.to_out.0",
        "transformers.{m}.blocks.{l}.cross_attention.query" =>
          "attentions.{m}.transformer_blocks.{l}.attn2.to_q",
        "transformers.{m}.blocks.{l}.cross_attention.key" =>
          "attentions.{m}.transformer_blocks.{l}.attn2.to_k",
        "transformers.{m}.blocks.{l}.cross_attention.value" =>
          "attentions.{m}.transformer_blocks.{l}.attn2.to_v",
        "transformers.{m}.blocks.{l}.cross_attention.output" =>
          "attentions.{m}.transformer_blocks.{l}.attn2.to_out.0",
        "transformers.{m}.blocks.{l}.ffn.intermediate" =>
          "attentions.{m}.transformer_blocks.{l}.ff.net.0.proj",
        "transformers.{m}.blocks.{l}.ffn.output" =>
          "attentions.{m}.transformer_blocks.{l}.ff.net.2",
        "transformers.{m}.blocks.{l}.self_attention_norm" =>
          "attentions.{m}.transformer_blocks.{l}.norm1",
        "transformers.{m}.blocks.{l}.cross_attention_norm" =>
          "attentions.{m}.transformer_blocks.{l}.norm2",
        "transformers.{m}.blocks.{l}.output_norm" =>
          "attentions.{m}.transformer_blocks.{l}.norm3",
        "residual_blocks.{m}.timestep_projection" => "resnets.{m}.time_emb_proj",
        "residual_blocks.{m}.norm_1" => "resnets.{m}.norm1",
        "residual_blocks.{m}.conv_1" => "resnets.{m}.conv1",
        "residual_blocks.{m}.norm_2" => "resnets.{m}.norm2",
        "residual_blocks.{m}.conv_2" => "resnets.{m}.conv2",
        "residual_blocks.{m}.shortcut.projection" => "resnets.{m}.conv_shortcut",
        "downsamples.{m}.conv" => "downsamplers.{m}.conv"
      }

      blocks_mapping =
        ["down_blocks.{n}", "mid_block"]
        |> Enum.map(&Transformers.Utils.prefix_params_mapping(block_mapping, &1, &1))
        |> Enum.reduce(&Map.merge/2)

      controlnet = %{
        "controlnet_conditioning_embedding.input_conv" => "controlnet_cond_embedding.conv_in",
        "controlnet_conditioning_embedding.inner_convs.{m}" =>
          "controlnet_cond_embedding.blocks.{m}",
        "controlnet_conditioning_embedding.output_conv" => "controlnet_cond_embedding.conv_out",
        "controlnet_down_blocks.{m}.zero_conv" => "controlnet_down_blocks.{m}",
        "controlnet_mid_block.zero_conv" => "controlnet_mid_block"
      }

      %{
        "time_embedding.intermediate" => "time_embedding.linear_1",
        "time_embedding.output" => "time_embedding.linear_2",
        "input_conv" => "conv_in"
      }
      |> Map.merge(blocks_mapping)
      |> Map.merge(controlnet)
    end
  end
end
