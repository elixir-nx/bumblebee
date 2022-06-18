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

    * `:layer_norm_init_value` - initial value for layer normalization scale.
      Defaults to `1.0e-6`

    * `:drop_path_rate` - drop path rate for stochastic depth. Defaults to
      `0.0`

  ### Common Options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}

  ## References

    * [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  """

  alias Bumblebee.Shared
  import Nx.Defn

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
    convnext = base_model(config, "convnext")

    outputs = Bumblebee.Utils.Axon.unwrap_container(convnext)

    logits =
      outputs.pooler_output
      |> Axon.dense(config.num_labels, name: "classifier")

    Axon.container(%{logits: logits, hidden_states: outputs.hidden_states})
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    base_model(config, "convnext")
  end

  defp base_model(config, name) do
    pixel_values = Axon.input({nil, config.num_channels, 224, 224}, "pixel_values")

    # TODO: Correct initializers

    embedding_output = embeddings(pixel_values, config, name: join(name, "embeddings"))

    {last_hidden_state, hidden_states} =
      encoder(embedding_output, config, name: join(name, "encoder"))

    pooled_output =
      last_hidden_state
      |> Axon.global_avg_pool()
      |> Axon.layer_norm(epsilon: config.layer_norm_eps, name: join(name, "layernorm"))

    Axon.container(%{
      last_hidden_state: last_hidden_state,
      pooler_output: pooled_output,
      hidden_states: hidden_states
    })
  end

  defp embeddings(%Axon{} = pixel_values, config, opts) do
    name = opts[:name]
    [embedding_size | _] = config.hidden_sizes

    pixel_values
    |> Axon.conv(embedding_size,
      kernel_size: config.patch_size,
      strides: config.patch_size,
      name: join(name, "patch_embeddings")
    )
    |> Axon.layer_norm(epsilon: 1.0e-6, name: join(name, "layernorm"))
  end

  defp encoder(hidden_states, config, opts) do
    name = opts[:name]

    drop_path_rates = get_drop_path_rates(config.depths, config.drop_path_rate)
    last_hidden_state = hidden_states
    all_hidden_states = {last_hidden_state}

    for idx <- 0..(config.num_stages - 1), reduce: {last_hidden_state, all_hidden_states} do
      {lhs, states} ->
        strides = if idx > 0, do: 2, else: 1
        depth = Enum.at(config.depths, idx)
        drop_path_rates = Enum.at(drop_path_rates, idx)
        out_channels = Enum.at(config.hidden_sizes, idx)

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
        |> Axon.layer_norm(epsilon: 1.0e-6, name: join(name, "downsampling_layer.0"))
        |> Axon.conv(out_channels,
          kernel_size: 2,
          strides: strides,
          name: join(name, "downsampling_layer.1")
        )
      else
        hidden_states
      end

    for idx <- 0..(depth - 1), reduce: downsampled do
      x ->
        layer_name = join("layers", "#{idx}")

        conv_next_layer(x, out_channels, config,
          name: join(name, layer_name),
          drop_path_rate: Enum.at(drop_path_rates, idx)
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
        name: join(name, "dwconv")
      )
      |> Axon.transpose([0, 2, 3, 1], ignore_batch?: false, name: join(name, "transpose1"))
      |> Axon.layer_norm(epsilon: 1.0e-6, channel_index: 3, name: join(name, "layernorm"))
      |> Axon.dense(4 * out_channels, name: join(name, "pwconv1"))
      |> Axon.activation(config.hidden_act, name: join(name, "activation"))
      |> Axon.dense(out_channels, name: join(name, "pwconv2"))

    scaled =
      if config.layer_scale_init_value > 0 do
        scale =
          Axon.param("layer_scale_parameter", {out_channels},
            initializer: fn shape, _ ->
              Nx.broadcast(config.layer_scale_init_value, shape)
            end
          )

        Axon.layer(fn x, y, _opts -> Nx.multiply(x, y) end, [x, scale],
          name: name,
          op_name: :scale
        )
      else
        x
      end

    scaled
    |> Axon.transpose([0, 3, 1, 2], ignore_batch?: false, name: join(name, "transpose2"))
    |> drop_path_layer(rate: drop_path_rate, name: join(name, "drop_path"))
    |> Axon.add(input, name: join(name, "residual"))
  end

  defp drop_path_layer(%Axon{} = input, opts) do
    opts = Keyword.validate!(opts, [:name, rate: 0.0])

    if opts[:rate] > 0.0 do
      Axon.layer(&drop_path/2, [input], name: opts[:name], rate: opts[:rate])
    else
      input
    end
  end

  defnp drop_path(x, opts \\ []) do
    opts = keyword!(opts, rate: 0.0, mode: :train)

    transform({x, opts[:rate], opts[:mode]}, fn
      {x, rate, :train} ->
        keep_prob = 1 - rate
        shape = elem(Nx.shape(x), 0)

        random_tensor =
          keep_prob
          |> Nx.add(Nx.random_uniform(shape))
          |> Nx.floor()

        out = x |> Nx.divide(keep_prob) |> Nx.multiply(random_tensor)
        # Do not apply if rate is 0.0
        if Elixir.Kernel.==(rate, 0.0) do
          x
        else
          out
        end

      {x, _rate, :inference} ->
        x
    end)
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
