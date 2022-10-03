defmodule Bumblebee.Vision.ResNet do
  @common_keys [:output_hidden_states, :id2label, :num_labels]

  @moduledoc """
  Models based on the ResNet architecture.

  ## Architectures

    * `:base` - plain ResNet without any head on top

    * `:for_image_classification` - ResNet with a classification head.
      The head consists of a single dense layer on top of the pooled
      features and it returns logits corresponding to possible classes

  ## Inputs

    * `"pixel_values"` - {batch_size, num_channels, height, width}

      Featurized image pixel values (224x224).

  ## Configuration

    * `:num_channels` - the number of input channels. Defaults to `3`

    * `:embedding_size` - dimensionality for the embedding layer.
      Defaults to `64`

    * `:hidden_sizes` - dimensionality at each stage. Defaults to
      `[256, 512, 1024, 2048]`

    * `:depths` - depth (number of layers) for each stage. Defaults
      to `[3, 4, 6, 3]`

    * `:layer_type` - the layer to use, either `:basic` (used for
      smaller models, like ResNet-18 or ResNet-34) or `:bottleneck`
      (used for larger models like ResNet-50 and above). Defaults to
      `:bottleneck`

    * `:hidden_act` - the activation function in each block. Defaults
      to `:relu`

    * `:downsample_in_first_stage` - whether the first stage should
      downsample the inputs using a stride of 2. Defaults to `false`

  ### Common options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              num_channels: 3,
              embedding_size: 64,
              hidden_sizes: [256, 512, 1024, 2048],
              depths: [3, 4, 6, 3],
              layer_type: :bottleneck,
              hidden_act: :relu,
              downsample_in_first_stage: false
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

  @impl true
  def architectures(), do: [:base, :for_image_classification]

  @impl true
  def base_model_prefix(), do: "resnet"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def input_template(config) do
    %{
      "pixel_values" => Nx.template({1, config.num_channels, 224, 224}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    config
    |> resnet()
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = config) do
    outputs = resnet(config, name: "resnet")

    logits =
      outputs.pooler_output
      |> Axon.flatten(name: "classifier.0")
      |> Axon.dense(config.num_labels, name: "classifier.1")

    Layers.output(%{logits: logits, hidden_states: outputs.hidden_states})
  end

  defp resnet(config, opts \\ []) do
    name = opts[:name]

    encoder_outputs =
      Axon.input("pixel_values", shape: {nil, config.num_channels, 224, 224})
      |> embedding_layer(config, name: join(name, "embedder"))
      |> encoder(config, name: join(name, "encoder"))

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

  defp embedding_layer(%Axon{} = hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> conv_layer(config.embedding_size,
      kernel_size: 7,
      strides: 2,
      activation: config.hidden_act,
      name: join(name, "embedder")
    )
    |> Axon.max_pool(
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "pooler")
    )
  end

  defp encoder(%Axon{} = hidden_state, config, opts) do
    name = opts[:name]

    stages = config.hidden_sizes |> Enum.zip(config.depths) |> Enum.with_index()

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, config.output_hidden_states),
      in_channels: config.embedding_size
    }

    for {{size, depth}, idx} <- stages, reduce: state do
      state ->
        strides = if idx == 0 and not config.downsample_in_first_stage, do: 1, else: 2

        hidden_state =
          stage(state.last_hidden_state, state.in_channels, size, config,
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

  defp stage(%Axon{} = hidden_state, in_channels, out_channels, config, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 2, depth: 2])
    name = opts[:name]
    strides = opts[:strides]
    depth = opts[:depth]

    layer =
      case config.layer_type do
        :bottleneck -> &bottleneck_layer/4
        :basic -> &basic_layer/4
      end

    # Downsampling is done in the first layer with stride of 2
    hidden_state =
      layer.(hidden_state, in_channels, out_channels,
        strides: strides,
        activation: config.hidden_act,
        name: join(name, "layers.0")
      )

    for idx <- 1..(depth - 1), reduce: hidden_state do
      hidden_state ->
        layer.(hidden_state, out_channels, out_channels,
          activation: config.hidden_act,
          name: join(name, "layers.#{idx}")
        )
    end
  end

  defp basic_layer(%Axon{} = hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]

    should_apply_shortcut? = in_channels != out_channels or strides != 1

    residual =
      if should_apply_shortcut? do
        shortcut_layer(hidden_state, out_channels, strides: strides, name: join(name, "shortcut"))
      else
        hidden_state
      end

    hidden_state
    |> conv_layer(out_channels, strides: strides, name: join(name, "layer.0"))
    |> conv_layer(out_channels, activation: :linear, name: join(name, "layer.1"))
    |> Axon.add(residual)
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp bottleneck_layer(%Axon{} = hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu, reduction: 4])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]
    reduction = opts[:reduction]

    should_apply_shortcut? = in_channels != out_channels or strides != 1
    reduces_channels = div(out_channels, reduction)

    residual =
      if should_apply_shortcut? do
        shortcut_layer(hidden_state, out_channels, strides: strides, name: join(name, "shortcut"))
      else
        hidden_state
      end

    hidden_state
    |> conv_layer(reduces_channels, kernel_size: 1, name: join(name, "layer.0"))
    |> conv_layer(reduces_channels, strides: strides, name: join(name, "layer.1"))
    |> conv_layer(out_channels, kernel_size: 1, activation: :linear, name: join(name, "layer.2"))
    |> Axon.add(residual)
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp shortcut_layer(%Axon{} = hidden_state, out_channels, opts) do
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

  defp conv_layer(%Axon{} = hidden_state, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, kernel_size: 3, strides: 1, activation: :relu])
    name = opts[:name]
    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    activation = opts[:activation]

    edge_padding = div(kernel_size, 2)
    padding_config = [{edge_padding, edge_padding}, {edge_padding, edge_padding}]

    hidden_state
    |> Axon.conv(out_channels,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding_config,
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
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["layer_type", "hidden_act"])
      |> Shared.convert_common()
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
