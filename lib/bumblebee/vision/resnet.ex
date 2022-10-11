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

    * `"pixel_values"` - {batch_size, num_channels, height, width}

      Featurized image pixel values (224x224).

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures(), do: [:base, :for_image_classification]

  @impl true
  def base_model_prefix(), do: "resnet"

  @impl true
  def config(spec, opts \\ []) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(spec) do
    %{
      "pixel_values" => Nx.template({1, spec.num_channels, 224, 224}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    spec
    |> resnet()
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    outputs = resnet(spec, name: "resnet")

    logits =
      outputs.pooler_output
      |> Axon.flatten(name: "classifier.0")
      |> Axon.dense(spec.num_labels, name: "classifier.1")

    Layers.output(%{logits: logits, hidden_states: outputs.hidden_states})
  end

  defp resnet(spec, opts \\ []) do
    name = opts[:name]

    encoder_outputs =
      Axon.input("pixel_values", shape: {nil, spec.num_channels, 224, 224})
      |> embeddings(spec, name: join(name, "embedder"))
      |> encoder(spec, name: join(name, "encoder"))

    pooled_output =
      Axon.adaptive_avg_pool(encoder_outputs.last_hidden_state,
        output_size: {1, 1},
        name: join(name, "pooler")
      )

    %{
      last_hidden_state: encoder_outputs.last_hidden_state,
      pooler_output: pooled_output,
      hidden_states: encoder_outputs.hidden_states
    }
  end

  defp embeddings(%Axon{} = hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> conv(spec.embedding_size,
      kernel_size: 7,
      strides: 2,
      activation: spec.activation,
      name: join(name, "embedder")
    )
    |> Axon.max_pool(
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "pooler")
    )
  end

  defp encoder(%Axon{} = hidden_state, spec, opts) do
    name = opts[:name]

    stages = spec.hidden_sizes |> Enum.zip(spec.depths) |> Enum.with_index()

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      in_channels: spec.embedding_size
    }

    for {{size, depth}, idx} <- stages, reduce: state do
      state ->
        strides = if idx == 0 and not spec.downsample_in_first_stage, do: 1, else: 2

        hidden_state =
          stage(state.last_hidden_state, state.in_channels, size, spec,
            depth: depth,
            strides: strides,
            name: join(name, "stages.#{idx}")
          )

        %{
          last_hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          in_channels: size
        }
    end
  end

  defp stage(%Axon{} = hidden_state, in_channels, out_channels, spec, opts) do
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
        name: join(name, "layers.0")
      )

    for idx <- 1..(depth - 1), reduce: hidden_state do
      hidden_state ->
        residual_block.(hidden_state, out_channels, out_channels,
          activation: spec.activation,
          name: join(name, "layers.#{idx}")
        )
    end
  end

  defp basic_residual_block(%Axon{} = hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]

    use_shortcut? = in_channels != out_channels or strides != 1

    residual =
      if use_shortcut? do
        shortcut(hidden_state, out_channels, strides: strides, name: join(name, "shortcut"))
      else
        hidden_state
      end

    hidden_state
    |> conv(out_channels, strides: strides, name: join(name, "layer.0"))
    |> conv(out_channels, activation: :linear, name: join(name, "layer.1"))
    |> Axon.add(residual)
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp bottleneck_residual_block(%Axon{} = hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu, reduction: 4])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]
    reduction = opts[:reduction]

    use_shortcut? = in_channels != out_channels or strides != 1
    reduced_channels = div(out_channels, reduction)

    residual =
      if use_shortcut? do
        shortcut(hidden_state, out_channels, strides: strides, name: join(name, "shortcut"))
      else
        hidden_state
      end

    hidden_state
    |> conv(reduced_channels, kernel_size: 1, name: join(name, "layer.0"))
    |> conv(reduced_channels, strides: strides, name: join(name, "layer.1"))
    |> conv(out_channels, kernel_size: 1, activation: :linear, name: join(name, "layer.2"))
    |> Axon.add(residual)
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp shortcut(%Axon{} = hidden_state, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 2])
    name = opts[:name]
    strides = opts[:strides]

    hidden_state
    |> Axon.conv(out_channels,
      kernel_size: 1,
      strides: strides,
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "convolution")
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "normalization"))
  end

  defp conv(%Axon{} = hidden_state, out_channels, opts) do
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
      name: join(name, "convolution")
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "normalization"))
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
          activation: {"hidden_act", atom()},
          downsample_in_first_stage: {"downsample_in_first_stage", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
