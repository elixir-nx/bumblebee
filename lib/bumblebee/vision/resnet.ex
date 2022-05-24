defmodule Bumblebee.Vision.ResNet do
  @moduledoc """
  ResNet-50 Architecture.
  """
  # TODO: These are the HF config defaults for ResNet config and should be
  # somehow included in the function which builds the model
  # @num_channels 3
  @embedding_size 64
  @hidden_sizes [256, 512, 1024, 2048]
  @depths [3, 4, 6, 3]
  @layer_type :bottleneck
  @hidden_act :relu
  @downsample_in_first_stage false

  def conv_layer(%Axon{} = x, out_channels, opts \\ []) do
    opts = Keyword.validate!(opts, kernel_size: 3, strides: 1, activation: :relu)
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
      use_bias: false
    )
    |> Axon.batch_norm()
    |> Axon.activation(activation)
  end

  def embedding_layer(%Axon{} = x) do
    x
    |> conv_layer(@embedding_size, kernel_size: 7, strides: 2, activation: @hidden_act)
    |> Axon.max_pool(kernel_size: 3, strides: 2, padding: [{1, 1}, {1, 1}])
  end

  def shortcut_layer(%Axon{} = x, out_channels, opts \\ []) do
    opts = Keyword.validate!(opts, strides: 2)
    strides = opts[:strides]

    x
    |> Axon.conv(out_channels, kernel_size: 1, strides: strides, use_bias: false)
    |> Axon.batch_norm()
  end

  def basic_layer(%Axon{} = x, out_channels, opts \\ []) do
    opts = Keyword.validate!(opts, strides: 1, activation: :relu)
    strides = opts[:strides]
    activation = opts[:activation]

    # init
    should_apply_shortcut = get_channels(x) != out_channels or strides != 1

    shortcut =
      if should_apply_shortcut do
        &shortcut_layer(&1, out_channels, strides: strides)
      else
        & &1
      end

    layer = fn x ->
      x
      |> conv_layer(out_channels, strides: strides)
      |> conv_layer(out_channels, activation: :linear)
    end

    activation = &Axon.activation(&1, activation)

    # forward
    residual = x
    hidden_state = layer.(x)
    residual = shortcut.(residual)
    hidden_state = Axon.add(hidden_state, residual)
    hidden_state = activation.(hidden_state)

    hidden_state
  end

  def bottleneck_layer(%Axon{} = x, out_channels, opts \\ []) do
    opts = Keyword.validate!(opts, strides: 1, activation: :relu, reduction: 4)
    strides = opts[:strides]
    activation = opts[:activation]
    reduction = opts[:reduction]

    # init
    should_apply_shortcut = get_channels(x) != out_channels or strides != 1
    reduces_channels = div(out_channels, reduction)

    shortcut =
      if should_apply_shortcut do
        &shortcut_layer(&1, out_channels, strides: strides)
      else
        & &1
      end

    layer = fn x ->
      x
      |> conv_layer(reduces_channels, kernel_size: 1)
      |> conv_layer(reduces_channels, strides: strides)
      |> conv_layer(out_channels, kernel_size: 1, activation: :linear)
    end

    activation = &Axon.activation(&1, activation)

    # forward
    residual = x
    hidden_state = layer.(x)
    residual = shortcut.(residual)
    hidden_state = Axon.add(hidden_state, residual)
    hidden_state = activation.(hidden_state)

    hidden_state
  end

  def stage(%Axon{} = x, out_channels, opts \\ []) do
    opts = Keyword.validate!(opts, strides: 2, depth: 2)
    strides = opts[:strides]
    depth = opts[:depth]

    # init
    layer =
      case @layer_type do
        :bottleneck ->
          &bottleneck_layer/3

        _ ->
          &basic_layer/3
      end

    # forward
    x = layer.(x, out_channels, strides: strides, activation: @hidden_act)

    for _ <- 1..(depth - 1), reduce: x do
      x ->
        layer.(x, out_channels, activation: @hidden_act)
    end
  end

  def encoder(%Axon{} = x) do
    # init
    [first_hidden_size | hidden_sizes] = @hidden_sizes
    [first_depth | depths] = @depths
    first_strides = if @downsample_in_first_stage, do: 2, else: 1

    first_stage = &stage(&1, first_hidden_size, strides: first_strides, depth: first_depth)

    rest_of_stages =
      for {size, depth} <- Enum.zip(hidden_sizes, depths) do
        &stage(&1, size, depth: depth)
      end

    stages = [first_stage | rest_of_stages]

    # forward
    for stage <- stages, reduce: x do
      hidden_state ->
        # TODO: output_hidden_states per HF model
        stage.(hidden_state)
    end
  end

  def model(%Axon{} = x) do
    # TODO: Correct initialization for layer types
    # init
    embedder = &embedding_layer/1
    encoder = &encoder/1
    pooler = &Axon.adaptive_avg_pool(&1, output_size: {1, 1})

    # forward
    embedding_output = embedder.(x)
    encoder_output = encoder.(embedding_output)
    pooled_output = pooler.(encoder_output)

    Axon.container(%{
      last_hidden_state: encoder_output,
      pooler_output: pooled_output
    })
  end

  defp get_channels(%Axon{output_shape: shape}) do
    # TODO: Should be flexible NCHW or NHWC
    elem(shape, 1)
  end
end
