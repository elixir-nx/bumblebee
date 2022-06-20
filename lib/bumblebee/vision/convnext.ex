defmodule Bumblebee.Vision.ConvNext do
  @common_keys [:id2label, :label2id, :num_labels, :output_hidden_states]

  @moduledoc """
  Model based on the ConvNeXT architecture.

  ## Architectures

    * `:base` - plain ConvNeXT without any head on top

    * `:for_image_classification` - ConvNeXT with a classification head.
      The head consists of a single dense layer on top of the pooled
      features

  ## Configuration

    * `:num_channels` - the number of input channels. Defaults to `3`

    * `:patch_size` - patch size to use in the embedding layer. Defaults
      to `4`

    * `:num_stages` - the number of stages of the model. Defaults to `4`

    * `:hidden_sizes` - dimensionality (hidden size) at each stage.
      Defaults to `[96, 192, 384, 768]`

    * `:depths` - depth (number of layers) for each stage. Defaults
      to `[3, 3, 9, 3]`

    * `:hidden_act` - the activation function in each block. Defaults
      to `:gelu`

    * `:initializer_range` - standard deviation of the truncated normal
      initializer for initializing weight matrices. Defaults to `0.02`

    * `:layer_norm_eps` - epsilon value used by layer normalization layers.
      Defaults to `1.0e-12`

    * `:layer_scale_init_value` - initial value for layer normalization scale.
      Defaults to `1.0e-6`

    * `:drop_path_rate` - drop path rate for stochastic depth. Defaults to
      `0.0`

  ### Common Options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}

  ## References

    * [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  """

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              num_channels: 3,
              patch_size: 4,
              num_stages: 4,
              hidden_sizes: [96, 192, 384, 768],
              depths: [3, 3, 9, 3],
              hidden_act: :gelu,
              initializer_range: 0.02,
              layer_norm_eps: 1.0e-12,
              layer_scale_init_value: 1.0e-6,
              drop_path_rate: 0.0
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

  @impl true
  def architectures(), do: [:base, :for_image_classification]

  @impl true
  def base_model_prefix(), do: "convnext"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def model(%__MODULE__{architecture: :for_image_classification} = config) do
    outputs = convnext(config, name: "convnext")

    logits =
      outputs.pooler_output
      |> Axon.dense(config.num_labels,
        name: "classifier",
        kernel_initializer: kernel_initializer(config)
      )

    Axon.container(%{logits: logits, hidden_states: outputs.hidden_states})
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    config
    |> convnext(name: "convnext")
    |> Axon.container()
  end

  defp convnext(config, opts) do
    name = opts[:name]

    pixel_values = Axon.input({nil, config.num_channels, 224, 224}, "pixel_values")

    embedding_output = embeddings(pixel_values, config, name: join(name, "embeddings"))

    {last_hidden_state, hidden_states} =
      encoder(embedding_output, config, name: join(name, "encoder"))

    pooled_output =
      last_hidden_state
      |> Axon.global_avg_pool()
      |> Axon.layer_norm(
        epsilon: config.layer_norm_eps,
        name: join(name, "layernorm"),
        beta_initializer: :zeros,
        gamma_initializer: :ones
      )

    %{
      last_hidden_state: last_hidden_state,
      pooler_output: pooled_output,
      hidden_states: hidden_states
    }
  end

  defp embeddings(%Axon{} = pixel_values, config, opts) do
    name = opts[:name]
    [embedding_size | _] = config.hidden_sizes

    pixel_values
    |> Axon.conv(embedding_size,
      kernel_size: config.patch_size,
      strides: config.patch_size,
      name: join(name, "patch_embeddings"),
      kernel_initializer: kernel_initializer(config)
    )
    |> Axon.layer_norm(
      epsilon: 1.0e-6,
      name: join(name, "layernorm"),
      beta_initializer: :zeros,
      gamma_initializer: :ones
    )
  end

  defp encoder(hidden_states, config, opts) do
    name = opts[:name]

    drop_path_rates = get_drop_path_rates(config.depths, config.drop_path_rate)
    last_hidden_state = hidden_states
    all_hidden_states = {last_hidden_state}

    stages =
      Enum.zip([0..(config.num_stages - 1), config.depths, drop_path_rates, config.hidden_sizes])

    for {idx, depth, drop_path_rates, out_channels} <- stages,
        reduce: {last_hidden_state, all_hidden_states} do
      {lhs, states} ->
        strides = if idx > 0, do: 2, else: 1

        stage_name = join("stages", "#{idx}")

        state =
          conv_next_stage(
            lhs,
            out_channels,
            config,
            strides: strides,
            depth: depth,
            drop_path_rates: drop_path_rates,
            name: join(name, stage_name)
          )

        {state, Tuple.append(states, state)}
    end
  end

  defp conv_next_stage(hidden_states, out_channels, config, opts) do
    name = opts[:name]

    strides = opts[:strides]
    depth = opts[:depth]
    drop_path_rates = opts[:drop_path_rates]

    in_channels = get_channels(hidden_states)

    downsampled =
      if in_channels != out_channels or strides > 1 do
        hidden_states
        |> Axon.layer_norm(
          epsilon: 1.0e-6,
          name: join(name, "downsampling_layer.0"),
          beta_initializer: :zeros,
          gamma_initializer: :ones
        )
        |> Axon.conv(out_channels,
          kernel_size: 2,
          strides: strides,
          name: join(name, "downsampling_layer.1"),
          kernel_initializer: kernel_initializer(config)
        )
      else
        hidden_states
      end

    # This is essentially the same as calling `with_index`, but
    # will ensure that we error out if for some reason we didn't
    # compute drop_path_rates right (e.g. it doesn't match the depth)
    for {drop_path_rate, idx} <- Enum.zip(drop_path_rates, 0..(depth - 1)), reduce: downsampled do
      x ->
        layer_name = join("layers", "#{idx}")

        conv_next_layer(x, out_channels, config,
          name: join(name, layer_name),
          drop_path_rate: drop_path_rate
        )
    end
  end

  defp conv_next_layer(%Axon{} = hidden_states, out_channels, config, opts) do
    name = opts[:name]

    drop_path_rate = opts[:drop_path_rate]

    input = hidden_states
    channel_multiplier = div(out_channels, get_channels(input))

    x =
      hidden_states
      |> Axon.depthwise_conv(channel_multiplier,
        kernel_size: 7,
        padding: [{3, 3}, {3, 3}],
        name: join(name, "dwconv"),
        kernel_initializer: kernel_initializer(config)
      )
      |> Axon.transpose([0, 2, 3, 1], ignore_batch?: false, name: join(name, "transpose1"))
      |> Axon.layer_norm(
        epsilon: 1.0e-6,
        channel_index: 3,
        name: join(name, "layernorm"),
        beta_initializer: :zeros,
        gamma_initializer: :ones
      )
      |> Axon.dense(4 * out_channels,
        name: join(name, "pwconv1"),
        kernel_initializer: kernel_initializer(config)
      )
      |> Axon.activation(config.hidden_act, name: join(name, "activation"))
      |> Axon.dense(out_channels,
        name: join(name, "pwconv2"),
        kernel_initializer: kernel_initializer(config)
      )

    scaled =
      if config.layer_scale_init_value > 0 do
        Layers.scale_layer(x,
          name: name,
          scale_init_value: config.layer_scale_init_value,
          channel_index: 3
        )
      else
        x
      end

    scaled
    |> Axon.transpose([0, 3, 1, 2], ignore_batch?: false, name: join(name, "transpose2"))
    |> Layers.drop_path_layer(rate: drop_path_rate, name: join(name, "drop_path"))
    |> Axon.add(input, name: join(name, "residual"))
  end

  defp get_channels(%Axon{output_shape: shape}) do
    elem(shape, 1)
  end

  defp get_drop_path_rates(depths, rate) do
    sum_of_depths = Enum.sum(depths)
    # It's a linspace from 0..sum_of_depths
    step = rate / (sum_of_depths - 1)
    rates = Nx.iota({sum_of_depths}) |> Nx.multiply(step)
    # Split so that we have same number of rates that match
    # each depth
    {_, final_rates} =
      for depth <- depths, reduce: {rates, []} do
        {rates, acc} ->
          rate_slice = rates[0..(depth - 1)//1]

          rates =
            if depth == Nx.size(rates) do
              rates
            else
              rates[depth..-1//1]
            end

          {rates, [Nx.to_flat_list(rate_slice) | acc]}
      end

    Enum.reverse(final_rates)
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.initializer_range)
  end

  defp join(lhs, rhs), do: lhs <> "." <> rhs

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.atomize_values(["hidden_act"])
      |> Shared.cast_common_values()
      |> Shared.data_into_config(config)
    end
  end
end
