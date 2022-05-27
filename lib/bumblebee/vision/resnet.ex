defmodule Bumblebee.Vision.ResNet do
  @moduledoc """
  ResNet architecture.
  """

  @behaviour Bumblebee.Architecture

  @doc """
  Model configuration.

  ## Options

    * `:num_channels` - the number of input channels. Defaults to `3`

    * `:embedding_size` - dimensionality (hidden size) for the
      embedding layer. Defaults to `64`

    * `:hidden_sizes` - dimensionality (hidden size) at each stage.
      Defaults to `[256, 512, 1024, 2048]`

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

  #{Bumblebee.Config.common_options_docs([:output_hidden_states, :id2label, :num_labels])}
  """
  @impl true
  def config(opts \\ []) do
    defaults = [
      num_channels: 3,
      embedding_size: 64,
      hidden_sizes: [256, 512, 1024, 2048],
      depths: [3, 4, 6, 3],
      layer_type: :bottleneck,
      hidden_act: :relu,
      downsample_in_first_stage: false
    ]

    common_keys = [:output_hidden_states, :id2label, :num_labels]

    Bumblebee.Config.build_config(opts, common_keys, defaults, atoms: [:layer_type, :hidden_act])
  end

  @impl true
  def base_model_prefix(), do: "resnet"

  @doc """
  Builds a ResNet model with a classification head.

  The classification head consists of a single dense layer on top of
  the pooled features.

  ## Options

    * `:config` - see `config/1`

  """
  def model_for_image_classification(opts \\ []) do
    config = Keyword.get_lazy(opts, :config, &config/0)

    resnet = model(config: config, name: "resnet")

    outputs = Bumblebee.Utils.Axon.unwrap_container(resnet)

    logits =
      outputs.pooler_output
      |> Axon.flatten(name: "classifier.0")
      |> Axon.dense(config.num_labels, name: "classifier.1")

    Axon.container(%{logits: logits, hidden_states: outputs.hidden_states})
  end

  @doc """
  Builds a ResNet model without any head.

  The model ends with a pooling layer.

  ## Options

    * `:config` - see `config/1`

    * `:name` - prefix for all layer names

  """
  def model(opts \\ []) do
    config = Keyword.get_lazy(opts, :config, &config/0)
    name = if(name = opts[:name], do: name <> ".", else: "")

    x = Axon.input({nil, config.num_channels, 224, 224})

    # TODO: Correct initialization for layer types
    embedder = &embedding_layer(&1, config, name: name <> "embedder")
    encoder = &encoder(&1, config, name: name <> "encoder")
    pooler = &Axon.adaptive_avg_pool(&1, output_size: {1, 1}, name: name <> "pooler")

    embedding_output = embedder.(x)
    {encoder_output, hidden_states} = encoder.(embedding_output)
    pooled_output = pooler.(encoder_output)

    Axon.container(%{
      last_hidden_state: encoder_output,
      pooler_output: pooled_output,
      hidden_states: if(config.output_hidden_states, do: hidden_states, else: {})
    })
  end

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

    [first_hidden_size | hidden_sizes] = config.hidden_sizes
    [first_depth | depths] = config.depths
    first_strides = if config.downsample_in_first_stage, do: 2, else: 1

    first_stage =
      &stage(&1, first_hidden_size, config,
        strides: first_strides,
        depth: first_depth,
        name: name <> ".stages.0"
      )

    rest_of_stages =
      for {{size, depth}, idx} <- Enum.zip(hidden_sizes, depths) |> Enum.with_index(1) do
        &stage(&1, size, config, depth: depth, name: name <> ".stages.#{idx}")
      end

    stages = [first_stage | rest_of_stages]

    for stage <- stages, reduce: {x, {}} do
      {hidden_state, hidden_states} ->
        {stage.(hidden_state), Tuple.append(hidden_states, hidden_state)}
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

    should_apply_shortcut = get_channels(x) != out_channels or strides != 1

    shortcut =
      if should_apply_shortcut do
        &shortcut_layer(&1, out_channels, strides: strides, name: name <> ".shortcut")
      else
        & &1
      end

    layer = fn x ->
      x
      |> conv_layer(out_channels, strides: strides, name: name <> ".layer.0")
      |> conv_layer(out_channels, activation: :linear, name: name <> ".layer.1")
    end

    activation = &Axon.activation(&1, activation, name: name <> ".activation")

    residual = x
    hidden_state = layer.(x)
    residual = shortcut.(residual)
    hidden_state = Axon.add(hidden_state, residual)
    hidden_state = activation.(hidden_state)

    hidden_state
  end

  defp bottleneck_layer(%Axon{} = x, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1, activation: :relu, reduction: 4])
    name = opts[:name]
    strides = opts[:strides]
    activation = opts[:activation]
    reduction = opts[:reduction]

    should_apply_shortcut = get_channels(x) != out_channels or strides != 1
    reduces_channels = div(out_channels, reduction)

    shortcut =
      if should_apply_shortcut do
        &shortcut_layer(&1, out_channels, strides: strides, name: name <> ".shortcut")
      else
        & &1
      end

    layer = fn x ->
      x
      |> conv_layer(reduces_channels, kernel_size: 1, name: name <> ".layer.0")
      |> conv_layer(reduces_channels, strides: strides, name: name <> ".layer.1")
      |> conv_layer(out_channels,
        kernel_size: 1,
        activation: :linear,
        name: name <> ".layer.2"
      )
    end

    activation = &Axon.activation(&1, activation, name: name <> ".activation")

    residual = x
    hidden_state = layer.(x)
    residual = shortcut.(residual)
    hidden_state = Axon.add(hidden_state, residual)
    hidden_state = activation.(hidden_state)

    hidden_state
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
      name: name <> ".convolution"
    )
    |> Axon.batch_norm(name: name <> ".normalization")
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
      name: name <> ".convolution"
    )
    |> Axon.batch_norm(name: name <> ".normalization")
    |> Axon.activation(activation, name: name <> ".activation")
  end

  defp get_channels(%Axon{output_shape: shape}) do
    # TODO: Should be flexible NCHW or NHWC
    elem(shape, 1)
  end
end
