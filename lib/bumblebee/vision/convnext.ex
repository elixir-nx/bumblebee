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
  ConvNeXT model family.

  ## Architectures

    * `:base` - plain ConvNeXT without any head on top

    * `:for_image_classification` - ConvNeXT with a classification head.
      The head consists of a single dense layer on top of the pooled
      features

  ## Inputs

    * `"pixel_values"` - {batch_size, height, width, num_channels}

      Featurized image pixel values (224x224).

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

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
      Axon.dense(outputs.pooled_state, spec.num_labels,
        name: "image_classification_head.output",
        kernel_initializer: kernel_initializer(spec)
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states
    })
  end

  defp core(spec, opts \\ []) do
    name = opts[:name]

    pixel_values = Axon.input("pixel_values", shape: {nil, 224, 224, spec.num_channels})

    encoder_outputs =
      pixel_values
      |> embedder(spec, name: join(name, "embedder"))
      |> encoder(spec, name: join(name, "encoder"))

    pooled_output =
      encoder_outputs.hidden_state
      |> Axon.global_avg_pool()
      |> Axon.layer_norm(
        name: join(name, "norm"),
        beta_initializer: :zeros,
        gamma_initializer: :ones
      )

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_output,
      hidden_states: encoder_outputs.hidden_states
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]
    embedding_size = hd(spec.hidden_sizes)

    pixel_values
    |> Axon.conv(embedding_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      name: join(name, "patch_embedding"),
      kernel_initializer: kernel_initializer(spec)
    )
    |> Axon.layer_norm(
      epsilon: 1.0e-6,
      name: join(name, "norm"),
      beta_initializer: :zeros,
      gamma_initializer: :ones
    )
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    drop_path_rates = get_drop_path_rates(spec.depths, spec.drop_path_rate)

    stages = Enum.zip([spec.depths, drop_path_rates, spec.hidden_sizes]) |> Enum.with_index()

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      in_channels: hd(spec.hidden_sizes)
    }

    for {{depth, drop_path_rates, out_channels}, idx} <- stages, reduce: state do
      state ->
        strides = if idx > 0, do: 2, else: 1

        hidden_state =
          stage(
            state.hidden_state,
            state.in_channels,
            out_channels,
            spec,
            strides: strides,
            depth: depth,
            drop_path_rates: drop_path_rates,
            name: name |> join("stages") |> join(idx)
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          in_channels: out_channels
        }
    end
  end

  defp stage(hidden_state, in_channels, out_channels, spec, opts) do
    name = opts[:name]

    strides = opts[:strides]
    depth = opts[:depth]
    drop_path_rates = opts[:drop_path_rates]

    downsampled =
      if in_channels != out_channels or strides > 1 do
        hidden_state
        |> Axon.layer_norm(
          epsilon: 1.0e-6,
          name: join(name, "downsample_norm"),
          beta_initializer: :zeros,
          gamma_initializer: :ones
        )
        |> Axon.conv(out_channels,
          kernel_size: 2,
          strides: strides,
          name: join(name, "downsample"),
          kernel_initializer: kernel_initializer(spec)
        )
      else
        hidden_state
      end

    # Ensure the rates have been computed properly
    ^depth = length(drop_path_rates)

    for {drop_path_rate, idx} <- Enum.with_index(drop_path_rates), reduce: downsampled do
      x ->
        block(x, out_channels, spec,
          name: name |> join("blocks") |> join(idx),
          drop_path_rate: drop_path_rate
        )
    end
  end

  defp block(hidden_state, out_channels, spec, opts) do
    name = opts[:name]
    drop_path_rate = opts[:drop_path_rate]

    input = hidden_state

    hidden_state =
      hidden_state
      |> Axon.depthwise_conv(1,
        kernel_size: 7,
        padding: [{3, 3}, {3, 3}],
        name: join(name, "depthwise_conv"),
        kernel_initializer: kernel_initializer(spec)
      )
      |> Axon.layer_norm(
        epsilon: 1.0e-6,
        channel_index: 3,
        name: join(name, "norm"),
        beta_initializer: :zeros,
        gamma_initializer: :ones
      )
      |> Axon.dense(4 * out_channels,
        name: join(name, "pointwise_conv_1"),
        kernel_initializer: kernel_initializer(spec)
      )
      |> Axon.activation(spec.activation, name: join(name, "activation"))
      |> Axon.dense(out_channels,
        name: join(name, "pointwise_conv_2"),
        kernel_initializer: kernel_initializer(spec)
      )

    hidden_state =
      if spec.scale_initial_value > 0 do
        Layers.scale(hidden_state,
          scale_initializer: Axon.Initializers.full(spec.scale_initial_value),
          name: join(name, "scale")
        )
      else
        hidden_state
      end

    hidden_state
    |> Layers.drop_path(rate: drop_path_rate, name: join(name, "drop_path"))
    |> Axon.add(input)
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

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          num_channels: {"num_channels", number()},
          patch_size: {"patch_size", number()},
          hidden_sizes: {"hidden_sizes", list(number())},
          depths: {"depths", list(number())},
          activation: {"hidden_act", activation()},
          scale_initial_value: {"layer_scale_init_value", number()},
          drop_path_rate: {"drop_path_rate", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding" => "convnext.embeddings.patch_embeddings",
        "embedder.norm" => "convnext.embeddings.layernorm",
        "encoder.stages.{n}.downsample_norm" =>
          "convnext.encoder.stages.{n}.downsampling_layer.0",
        "encoder.stages.{n}.downsample" => "convnext.encoder.stages.{n}.downsampling_layer.1",
        "encoder.stages.{n}.blocks.{m}.depthwise_conv" =>
          "convnext.encoder.stages.{n}.layers.{m}.dwconv",
        "encoder.stages.{n}.blocks.{m}.norm" =>
          "convnext.encoder.stages.{n}.layers.{m}.layernorm",
        "encoder.stages.{n}.blocks.{m}.pointwise_conv_1" =>
          "convnext.encoder.stages.{n}.layers.{m}.pwconv1",
        "encoder.stages.{n}.blocks.{m}.pointwise_conv_2" =>
          "convnext.encoder.stages.{n}.layers.{m}.pwconv2",
        "encoder.stages.{n}.blocks.{m}.scale" => %{
          "scale" => {
            [{"convnext.encoder.stages.{n}.layers.{m}", "layer_scale_parameter"}],
            fn [scale] -> scale end
          }
        },
        "norm" => "convnext.layernorm",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end
