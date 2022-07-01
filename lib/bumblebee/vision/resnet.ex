defmodule Bumblebee.Vision.ResNet do
  @common_keys [:output_hidden_states, :id2label, :label2id, :num_labels]

  @moduledoc """
  Models based on the ResNet architecture.

  ## Architectures

    * `:base` - plain ResNet without any head on top

    * `:for_image_classification` - ResNet with a classification head.
      The head consists of a single dense layer on top of the pooled
      features and it returns logits corresponding to possible classes

  ## Inputs

    * `"pixel_values"` - featurized image pixel values in NCHW format (224x224)

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

  alias Bumblebee.Shared

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
  def model(%__MODULE__{architecture: :base} = config) do
    config
    |> resnet()
    |> Axon.container()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = config) do
    outputs = resnet(config, name: "resnet")

    logits =
      outputs.pooler_output
      |> Axon.flatten(name: "classifier.0")
      |> Axon.dense(config.num_labels, name: "classifier.1")

    Axon.container(%{logits: logits, hidden_states: outputs.hidden_states})
  end

  defp resnet(config, opts \\ []) do
    name = opts[:name]

    {encoder_output, hidden_states} =
      Axon.input({nil, config.num_channels, 224, 224}, "pixel_values")
      |> embedding_layer(config, name: join(name, "embedder"))
      |> encoder(config, name: join(name, "encoder"))

    pooled_output =
      Axon.adaptive_avg_pool(encoder_output, output_size: {1, 1}, name: join(name, "pooler"))

    %{
      last_hidden_state: encoder_output,
      pooler_output: pooled_output,
      hidden_states: if(config.output_hidden_states, do: hidden_states, else: {})
    }
  end

  defp join(nil, suffix), do: suffix
  defp join(base, suffix), do: base <> "." <> suffix

  defp embedding_layer(%Axon{} = x, config, opts) do
    name = opts[:name]

    x
    |> conv_layer(config.embedding_size,
      kernel_size: 7,
      strides: 2,
      activation: config.hidden_act,
      name: name <> ".embedder"
    )
    |> Axon.max_pool(
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      name: name <> ".pooler"
    )
  end

  defp encoder(%Axon{} = x, config, opts) do
    name = opts[:name]

    stages = config.hidden_sizes |> Enum.zip(config.depths) |> Enum.with_index()

    for {{size, depth}, idx} <- stages, reduce: {x, {x}} do
      {x, hidden_states} ->
        strides = if idx == 0 and not config.downsample_in_first_stage, do: 1, else: 2
        x = stage(x, size, config, depth: depth, strides: strides, name: name <> ".stages.#{idx}")
        {x, Tuple.append(hidden_states, x)}
    end
  end

  defp stage(%Axon{} = x, out_channels, config, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 2, depth: 2])
    name = opts[:name]
    strides = opts[:strides]
    depth = opts[:depth]

    layer =
      case config.layer_type do
        :bottleneck -> &bottleneck_layer/3
        :basic -> &basic_layer/3
      end

    x =
      layer.(x, out_channels,
        strides: strides,
        activation: config.hidden_act,
        name: name <> ".layers.0"
      )

    for idx <- 1..(depth - 1), reduce: x do
      x ->
        layer.(x, out_channels, activation: config.hidden_act, name: name <> ".layers.#{idx}")
    end
  end

  defp basic_layer(%Axon{} = x, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]

    should_apply_shortcut? = get_channels(x) != out_channels or strides != 1

    residual =
      if should_apply_shortcut? do
        shortcut_layer(x, out_channels, strides: strides, name: name <> ".shortcut")
      else
        x
      end

    x
    |> conv_layer(out_channels, strides: strides, name: name <> ".layer.0")
    |> conv_layer(out_channels, activation: :linear, name: name <> ".layer.1")
    |> Axon.add(residual)
    |> Axon.activation(activation, name: name <> ".activation")
  end

  defp bottleneck_layer(%Axon{} = x, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu, reduction: 4])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]
    reduction = opts[:reduction]

    should_apply_shortcut? = get_channels(x) != out_channels or strides != 1
    reduces_channels = div(out_channels, reduction)

    residual =
      if should_apply_shortcut? do
        shortcut_layer(x, out_channels, strides: strides, name: name <> ".shortcut")
      else
        x
      end

    x
    |> conv_layer(reduces_channels, kernel_size: 1, name: name <> ".layer.0")
    |> conv_layer(reduces_channels, strides: strides, name: name <> ".layer.1")
    |> conv_layer(out_channels, kernel_size: 1, activation: :linear, name: name <> ".layer.2")
    |> Axon.add(residual)
    |> Axon.activation(activation, name: name <> ".activation")
  end

  defp shortcut_layer(%Axon{} = x, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 2])
    name = opts[:name]
    strides = opts[:strides]

    x
    |> Axon.conv(out_channels,
      kernel_size: 1,
      strides: strides,
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: name <> ".convolution"
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: name <> ".normalization")
  end

  defp conv_layer(%Axon{} = x, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, kernel_size: 3, strides: 1, activation: :relu])
    name = opts[:name]
    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    activation = opts[:activation]

    edge_padding = div(kernel_size, 2)
    padding_config = [{edge_padding, edge_padding}, {edge_padding, edge_padding}]

    x
    |> Axon.conv(out_channels,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding_config,
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: name <> ".convolution"
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: name <> ".normalization")
    |> Axon.activation(activation, name: name <> ".activation")
  end

  defp conv_kernel_initializer() do
    Axon.Initializers.variance_scaling(scale: 2.0, mode: :fan_out, distribution: :normal)
  end

  defp get_channels(%Axon{output_shape: shape}), do: elem(shape, 1)

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["layer_type", "hidden_act"])
      |> Shared.convert_common()
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
