defmodule Bumblebee.Vision.ResNet do
  alias Bumblebee.Shared

  options =
    [
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      embedding_size: [
        default: 64,
        doc: "the dimensionality of the embedding layer"
      ],
      hidden_sizes: [
        default: [256, 512, 1024, 2048],
        doc: "the dimensionality of hidden layers at each stage"
      ],
      depths: [
        default: [3, 4, 6, 3],
        doc: "the depth (number of residual blocks) at each stage"
      ],
      residual_block_type: [
        default: :bottleneck,
        doc: """
        the residual block to use, either `:basic` (used for smaller models, like ResNet-18 or ResNet-34)
        or `:bottleneck` (used for larger models like ResNet-50 and above)
        """
      ],
      activation: [
        default: :relu,
        doc: "the activation function"
      ],
      downsample_in_first_stage: [
        default: false,
        doc: "whether the first stage should downsample the inputs using a stride of 2"
      ]
    ] ++ Shared.common_options([:output_hidden_states, :num_labels, :id_to_label])

  @moduledoc """
  ResNet model family.

  ## Architectures

    * `:base` - plain ResNet without any head on top

    * `:for_image_classification` - ResNet with a classification head.
      The head consists of a single dense layer on top of the pooled
      features and it returns logits corresponding to possible classes

  ## Inputs

    * `"pixel_values"` - {batch_size, height, width, num_channels}

      Featurized image pixel values (224x224).

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_image_classification]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(spec) do
    %{
      "pixel_values" => Nx.template({1, 224, 224, spec.num_channels}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    spec
    |> core()
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    outputs = core(spec)

    logits =
      Axon.dense(outputs.pooled_state, spec.num_labels, name: "image_classification_head.output")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states
    })
  end

  defp core(spec, opts \\ []) do
    name = opts[:name]

    input = Axon.input("pixel_values", shape: {nil, 224, 224, spec.num_channels})

    encoder_outputs =
      input
      |> embedder(spec, name: join(name, "embedder"))
      |> encoder(spec, name: join(name, "encoder"))

    pooled_output =
      encoder_outputs.hidden_state
      |> Axon.adaptive_avg_pool(
        output_size: {1, 1},
        name: join(name, "pooler")
      )
      |> Axon.flatten()

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_output,
      hidden_states: encoder_outputs.hidden_states
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> conv_block(spec.embedding_size,
      kernel_size: 7,
      strides: 2,
      activation: spec.activation,
      name: join(name, "conv_block")
    )
    |> Axon.max_pool(
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "pooler")
    )
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    stages = spec.hidden_sizes |> Enum.zip(spec.depths) |> Enum.with_index()

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      in_channels: spec.embedding_size
    }

    for {{size, depth}, idx} <- stages, reduce: state do
      state ->
        strides = if idx == 0 and not spec.downsample_in_first_stage, do: 1, else: 2

        hidden_state =
          stage(state.hidden_state, state.in_channels, size, spec,
            depth: depth,
            strides: strides,
            name: join(name, "stages.#{idx}")
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          in_channels: size
        }
    end
  end

  defp stage(hidden_state, in_channels, out_channels, spec, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 2, depth: 2])
    name = opts[:name]
    strides = opts[:strides]
    depth = opts[:depth]

    residual_block =
      case spec.residual_block_type do
        :basic -> &basic_residual_block/4
        :bottleneck -> &bottleneck_residual_block/4
      end

    # Downsampling is done in the first block with stride of 2
    hidden_state =
      residual_block.(hidden_state, in_channels, out_channels,
        strides: strides,
        activation: spec.activation,
        name: join(name, "blocks.0")
      )

    for idx <- 1..(depth - 1)//1, reduce: hidden_state do
      hidden_state ->
        residual_block.(hidden_state, out_channels, out_channels,
          activation: spec.activation,
          name: join(name, "blocks.#{idx}")
        )
    end
  end

  defp basic_residual_block(hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]

    shortcut =
      shortcut(hidden_state, in_channels, out_channels,
        strides: strides,
        name: join(name, "shortcut")
      )

    hidden_state
    |> conv_block(out_channels, strides: strides, name: join(name, "conv_blocks.0"))
    |> conv_block(out_channels, activation: :linear, name: join(name, "conv_blocks.1"))
    |> Axon.add(shortcut)
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp bottleneck_residual_block(hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu, reduction: 4])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]
    reduction = opts[:reduction]

    shortcut =
      shortcut(hidden_state, in_channels, out_channels,
        strides: strides,
        name: join(name, "shortcut")
      )

    reduced_channels = div(out_channels, reduction)

    hidden_state
    |> conv_block(reduced_channels, kernel_size: 1, name: join(name, "conv_blocks.0"))
    |> conv_block(reduced_channels, strides: strides, name: join(name, "conv_blocks.1"))
    |> conv_block(out_channels,
      kernel_size: 1,
      activation: :linear,
      name: join(name, "conv_blocks.2")
    )
    |> Axon.add(shortcut)
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp shortcut(hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 2])
    name = opts[:name]
    strides = opts[:strides]

    # If the output shape doesn't match input shape, we need to project
    # the shortcut connection
    project_shortcut? = in_channels != out_channels or strides != 1

    if project_shortcut? do
      hidden_state
      |> Axon.conv(out_channels,
        kernel_size: 1,
        strides: strides,
        use_bias: false,
        kernel_initializer: conv_kernel_initializer(),
        name: join(name, "projection")
      )
      |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "norm"))
    else
      hidden_state
    end
  end

  defp conv_block(hidden_state, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, kernel_size: 3, strides: 1, activation: :relu])
    name = opts[:name]
    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    activation = opts[:activation]

    edge_padding = div(kernel_size, 2)
    padding_spec = [{edge_padding, edge_padding}, {edge_padding, edge_padding}]

    hidden_state
    |> Axon.conv(out_channels,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding_spec,
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "conv")
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "norm"))
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp conv_kernel_initializer() do
    Axon.Initializers.variance_scaling(scale: 2.0, mode: :fan_out, distribution: :normal)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          num_channels: {"num_channels", number()},
          embedding_size: {"embedding_size", number()},
          hidden_sizes: {"hidden_sizes", list(number())},
          depths: {"depths", list(number())},
          residual_block_type: {"layer_type", atom()},
          activation: {"hidden_act", activation()},
          downsample_in_first_stage: {"downsample_in_first_stage", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.conv_block.conv" => "resnet.embedder.embedder.convolution",
        "embedder.conv_block.norm" => "resnet.embedder.embedder.normalization",
        "encoder.stages.{n}.blocks.{m}.conv_blocks.{l}.conv" =>
          "resnet.encoder.stages.{n}.layers.{m}.layer.{l}.convolution",
        "encoder.stages.{n}.blocks.{m}.conv_blocks.{l}.norm" =>
          "resnet.encoder.stages.{n}.layers.{m}.layer.{l}.normalization",
        "encoder.stages.{n}.blocks.{m}.shortcut.projection" =>
          "resnet.encoder.stages.{n}.layers.{m}.shortcut.convolution",
        "encoder.stages.{n}.blocks.{m}.shortcut.norm" =>
          "resnet.encoder.stages.{n}.layers.{m}.shortcut.normalization",
        "image_classification_head.output" => "classifier.1"
      }
    end
  end
end
