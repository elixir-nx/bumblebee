defmodule Bumblebee.Vision.ConvNext do
  alias Bumblebee.Shared

  options =
    [
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 4,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_sizes: [
        default: [96, 192, 384, 768],
        doc: "the dimensionality of hidden layers at each stage"
      ],
      depths: [
        default: [3, 3, 9, 3],
        doc: "the depth (number of residual blocks) at each stage"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      scale_initial_value: [
        default: 1.0e-6,
        doc: "the initial value for scaling layers"
      ],
      drop_path_rate: [
        default: 0.0,
        doc: "the drop path rate used to for stochastic depth"
      ],
      layer_norm_epsilon: [
        default: 1.0e-12,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++ Shared.common_options([:output_hidden_states, :num_labels, :id_to_label])

  @moduledoc """
  Models based on the ConvNeXT architecture.

  ## Architectures

    * `:base` - plain ConvNeXT without any head on top

    * `:for_image_classification` - ConvNeXT with a classification head.
      The head consists of a single dense layer on top of the pooled
      features

  ## Inputs

    * `"pixel_values"` - {batch_size, num_channels, height, width}

      Featurized image pixel values (224x224).

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures(), do: [:base, :for_image_classification]

  @impl true
  def base_model_prefix(), do: "convnext"

  @impl true
  def config(featurizer, opts \\ []) do
    featurizer
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(featurizer) do
    %{
      "pixel_values" => Nx.template({1, featurizer.num_channels, 224, 224}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = featurizer) do
    featurizer
    |> convnext()
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = featurizer) do
    outputs = convnext(featurizer, name: "convnext")

    logits =
      outputs.pooler_output
      |> Axon.dense(featurizer.num_labels,
        name: "classifier",
        kernel_initializer: kernel_initializer(featurizer)
      )

    Layers.output(%{logits: logits, hidden_states: outputs.hidden_states})
  end

  defp convnext(featurizer, opts \\ []) do
    name = opts[:name]

    pixel_values = Axon.input("pixel_values", shape: {nil, featurizer.num_channels, 224, 224})

    embedding_output = embeddings(pixel_values, featurizer, name: join(name, "embeddings"))

    encoder_output = encoder(embedding_output, featurizer, name: join(name, "encoder"))

    pooled_output =
      encoder_output.last_hidden_state
      |> Axon.global_avg_pool()
      |> Axon.layer_norm(
        epsilon: featurizer.layer_norm_epsilon,
        name: join(name, "layernorm"),
        beta_initializer: :zeros,
        gamma_initializer: :ones
      )

    %{
      last_hidden_state: encoder_output.last_hidden_state,
      pooler_output: pooled_output,
      hidden_states: encoder_output.hidden_states
    }
  end

  defp embeddings(%Axon{} = pixel_values, featurizer, opts) do
    name = opts[:name]
    [embedding_size | _] = featurizer.hidden_sizes

    pixel_values
    |> Axon.conv(embedding_size,
      kernel_size: featurizer.patch_size,
      strides: featurizer.patch_size,
      name: join(name, "patch_embeddings"),
      kernel_initializer: kernel_initializer(featurizer)
    )
    |> Axon.layer_norm(
      epsilon: 1.0e-6,
      name: join(name, "layernorm"),
      beta_initializer: :zeros,
      gamma_initializer: :ones
    )
  end

  defp encoder(hidden_state, featurizer, opts) do
    name = opts[:name]

    drop_path_rates = get_drop_path_rates(featurizer.depths, featurizer.drop_path_rate)

    stages =
      Enum.zip([featurizer.depths, drop_path_rates, featurizer.hidden_sizes]) |> Enum.with_index()

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, featurizer.output_hidden_states),
      in_channels: hd(featurizer.hidden_sizes)
    }

    for {{depth, drop_path_rates, out_channels}, idx} <- stages, reduce: state do
      state ->
        strides = if idx > 0, do: 2, else: 1

        hidden_state =
          stage(
            state.last_hidden_state,
            state.in_channels,
            out_channels,
            featurizer,
            strides: strides,
            depth: depth,
            drop_path_rates: drop_path_rates,
            name: name |> join("stages") |> join(idx)
          )

        %{
          last_hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          in_channels: out_channels
        }
    end
  end

  defp stage(hidden_state, in_channels, out_channels, featurizer, opts) do
    name = opts[:name]

    strides = opts[:strides]
    depth = opts[:depth]
    drop_path_rates = opts[:drop_path_rates]

    downsampled =
      if in_channels != out_channels or strides > 1 do
        hidden_state
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
          kernel_initializer: kernel_initializer(featurizer)
        )
      else
        hidden_state
      end

    # Ensure the rates have been computed properly
    ^depth = length(drop_path_rates)

    for {drop_path_rate, idx} <- Enum.with_index(drop_path_rates), reduce: downsampled do
      x ->
        block(x, out_channels, featurizer,
          name: name |> join("layers") |> join(idx),
          drop_path_rate: drop_path_rate
        )
    end
  end

  defp block(%Axon{} = hidden_state, out_channels, featurizer, opts) do
    name = opts[:name]

    drop_path_rate = opts[:drop_path_rate]

    input = hidden_state

    x =
      hidden_state
      |> Axon.depthwise_conv(1,
        kernel_size: 7,
        padding: [{3, 3}, {3, 3}],
        name: join(name, "dwconv"),
        kernel_initializer: kernel_initializer(featurizer)
      )
      |> Axon.transpose([0, 2, 3, 1], name: join(name, "transpose1"))
      |> Axon.layer_norm(
        epsilon: 1.0e-6,
        channel_index: 3,
        name: join(name, "layernorm"),
        beta_initializer: :zeros,
        gamma_initializer: :ones
      )
      |> Axon.dense(4 * out_channels,
        name: join(name, "pwconv1"),
        kernel_initializer: kernel_initializer(featurizer)
      )
      |> Axon.activation(featurizer.activation, name: join(name, "activation"))
      |> Axon.dense(out_channels,
        name: join(name, "pwconv2"),
        kernel_initializer: kernel_initializer(featurizer)
      )

    scaled =
      if featurizer.scale_initial_value > 0 do
        Layers.scale(x,
          name: name,
          scale_initializer: Axon.Initializers.full(featurizer.scale_initial_value),
          scale_name: "layer_scale_parameter",
          channel_index: 3
        )
      else
        x
      end

    scaled
    |> Axon.transpose([0, 3, 1, 2], name: join(name, "transpose2"))
    |> Layers.drop_path(rate: drop_path_rate, name: join(name, "drop_path"))
    |> Axon.add(input, name: join(name, "residual"))
  end

  defp get_drop_path_rates(depths, rate) do
    sum_of_depths = Enum.sum(depths)

    rates =
      Nx.iota({sum_of_depths})
      |> Nx.multiply(rate / sum_of_depths - 1)
      |> Nx.to_flat_list()

    {final_rates, _} =
      Enum.map_reduce(depths, rates, fn depth, rates ->
        Enum.split(rates, depth)
      end)

    final_rates
  end

  defp kernel_initializer(featurizer) do
    Axon.Initializers.normal(scale: featurizer.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(featurizer, data) do
      import Shared.Converters

      opts =
        convert!(data,
          num_channels: {"num_channels", number()},
          patch_size: {"patch_size", number()},
          hidden_sizes: {"hidden_sizes", list(number())},
          depths: {"depths", list(number())},
          activation: {"hidden_act", atom()},
          scale_initial_value: {"layer_scale_init_value", number()},
          drop_path_rate: {"drop_path_rate", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, featurizer)

      @for.config(featurizer, opts)
    end
  end
end
