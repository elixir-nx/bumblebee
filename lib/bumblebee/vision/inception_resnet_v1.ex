defmodule Bumblebee.Vision.InceptionResnetV1 do
  alias Bumblebee.Shared

  options =
    [
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      image_size: [
        default: 160,
        doc: "the size of the input spatial dimensions"
      ],
      dropout_prob: [
        default: 0.6,
        doc: "the dropout probability"
      ],
      embedding_size: [
        default: 512,
        doc: "the dimensionality of the output embeddings"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  Inception-ResNet-V1 model family.

  ## Architectures

    * `:base` - plain InceptionResnetV1 without any head on top

    * `:for_image_classification` - InceptionResnetV1 with a classification head.
      The head consists of a single dense layer on top of the pooled
      features and it returns logits corresponding to possible classes

  ## Inputs

    * `"pixel_values"` - `{batch_size, height, width, num_channels}`

      Featurized image pixel values (160x160).

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
    * [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

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
      "pixel_values" => Nx.template({1, spec.image_size, spec.image_size, spec.num_channels}, :f32)
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

    input = Axon.input("pixel_values", shape: {nil, spec.image_size, spec.image_size, spec.num_channels})

    pooled_state =
      input
      |> stem(spec, name: join(name, "stem"))
      |> inception_resnet_blocks(spec, name: join(name, "blocks"))
      |> head(spec, name: join(name, "head"))

    %{
      pooled_state: pooled_state,
      hidden_states: Axon.container({input, pooled_state})
    }
  end

  # Stem: Initial convolutional layers
  defp stem(pixel_values, _spec, opts) do
    name = opts[:name]

    pixel_values
    |> basic_conv2d(32, kernel_size: 3, strides: 2, name: join(name, "conv2d_1a"))
    |> basic_conv2d(32, kernel_size: 3, strides: 1, name: join(name, "conv2d_2a"))
    |> basic_conv2d(64, kernel_size: 3, strides: 1, padding: [{1, 1}, {1, 1}], name: join(name, "conv2d_2b"))
    |> Axon.max_pool(kernel_size: 3, strides: 2, name: join(name, "maxpool_3a"))
    |> basic_conv2d(80, kernel_size: 1, strides: 1, name: join(name, "conv2d_3b"))
    |> basic_conv2d(192, kernel_size: 3, strides: 1, name: join(name, "conv2d_4a"))
    |> basic_conv2d(256, kernel_size: 3, strides: 2, name: join(name, "conv2d_4b"))
  end

  # Inception-ResNet blocks
  defp inception_resnet_blocks(hidden_state, _spec, opts) do
    name = opts[:name]

    hidden_state
    # 5x Block35 (Inception-ResNet-A)
    |> then(&Enum.reduce(0..4, &1, fn i, acc ->
      block35(acc, 0.17, name: join(name, "repeat_1.#{i}"))
    end))
    # Mixed 6a (Reduction-A)
    |> mixed_6a(name: join(name, "mixed_6a"))
    # 10x Block17 (Inception-ResNet-B)
    |> then(&Enum.reduce(0..9, &1, fn i, acc ->
      block17(acc, 0.10, name: join(name, "repeat_2.#{i}"))
    end))
    # Mixed 7a (Reduction-B)
    |> mixed_7a(name: join(name, "mixed_7a"))
    # 5x Block8 (Inception-ResNet-C)
    |> then(&Enum.reduce(0..4, &1, fn i, acc ->
      block8(acc, 0.20, name: join(name, "repeat_3.#{i}"))
    end))
    # Final Block8 (scale=1.0, no activation)
    |> block8(1.0, activation: :linear, name: join(name, "block8"))
  end

  # Head: pooling, dropout, and embedding projection
  defp head(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.adaptive_avg_pool(output_size: {1, 1}, name: join(name, "avgpool_1a"))
    |> Axon.flatten()
    |> Axon.dropout(rate: spec.dropout_prob, name: join(name, "dropout"))
    |> Axon.dense(spec.embedding_size, use_bias: false, name: join(name, "last_linear"))
    |> Axon.batch_norm(
      epsilon: 0.001,
      momentum: 0.1,
      gamma_initializer: :ones,
      name: join(name, "last_bn")
    )
  end

  # BasicConv2d: Conv2d + BatchNorm + ReLU
  defp basic_conv2d(x, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, kernel_size: 3, strides: 1, padding: :valid])
    name = opts[:name]
    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]

    x
    |> Axon.conv(out_channels,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding,
      use_bias: false,
      name: join(name, "conv")
    )
    |> Axon.batch_norm(
      epsilon: 0.001,
      momentum: 0.1,
      gamma_initializer: :ones,
      name: join(name, "bn")
    )
    |> Axon.activation(:relu, name: join(name, "relu"))
  end

  # Block35: Inception-ResNet-A
  defp block35(x, scale, opts) do
    name = opts[:name]

    branch0 = basic_conv2d(x, 32, kernel_size: 1, strides: 1, name: join(name, "branch0"))

    branch1 =
      x
      |> basic_conv2d(32, kernel_size: 1, strides: 1, name: join(name, "branch1.0"))
      |> basic_conv2d(32, kernel_size: 3, strides: 1, padding: [{1, 1}, {1, 1}], name: join(name, "branch1.1"))

    branch2 =
      x
      |> basic_conv2d(32, kernel_size: 1, strides: 1, name: join(name, "branch2.0"))
      |> basic_conv2d(32, kernel_size: 3, strides: 1, padding: [{1, 1}, {1, 1}], name: join(name, "branch2.1"))
      |> basic_conv2d(32, kernel_size: 3, strides: 1, padding: [{1, 1}, {1, 1}], name: join(name, "branch2.2"))

    Axon.concatenate([branch0, branch1, branch2], axis: 3)
    |> Axon.conv(256, kernel_size: 1, strides: 1, use_bias: true, name: join(name, "conv2d"))
    |> Axon.nx(fn conv -> Nx.multiply(conv, scale) end)
    |> Axon.add(x)
    |> Axon.activation(:relu, name: join(name, "relu"))
  end

  # Block17: Inception-ResNet-B
  defp block17(x, scale, opts) do
    name = opts[:name]

    branch0 = basic_conv2d(x, 128, kernel_size: 1, strides: 1, name: join(name, "branch0"))

    branch1 =
      x
      |> basic_conv2d(128, kernel_size: 1, strides: 1, name: join(name, "branch1.0"))
      |> basic_conv2d(128, kernel_size: {1, 7}, strides: 1, padding: [{0, 0}, {3, 3}], name: join(name, "branch1.1"))
      |> basic_conv2d(128, kernel_size: {7, 1}, strides: 1, padding: [{3, 3}, {0, 0}], name: join(name, "branch1.2"))

    Axon.concatenate([branch0, branch1], axis: 3)
    |> Axon.conv(896, kernel_size: 1, strides: 1, use_bias: true, name: join(name, "conv2d"))
    |> Axon.nx(fn conv -> Nx.multiply(conv, scale) end)
    |> Axon.add(x)
    |> Axon.activation(:relu, name: join(name, "relu"))
  end

  # Block8: Inception-ResNet-C
  defp block8(x, scale, opts) do
    opts = Keyword.validate!(opts, [:name, activation: :relu])
    name = opts[:name]
    activation = opts[:activation]

    branch0 = basic_conv2d(x, 192, kernel_size: 1, strides: 1, name: join(name, "branch0"))

    branch1 =
      x
      |> basic_conv2d(192, kernel_size: 1, strides: 1, name: join(name, "branch1.0"))
      |> basic_conv2d(192, kernel_size: {1, 3}, strides: 1, padding: [{0, 0}, {1, 1}], name: join(name, "branch1.1"))
      |> basic_conv2d(192, kernel_size: {3, 1}, strides: 1, padding: [{1, 1}, {0, 0}], name: join(name, "branch1.2"))

    residual =
      Axon.concatenate([branch0, branch1], axis: 3)
      |> Axon.conv(1792, kernel_size: 1, strides: 1, use_bias: true, name: join(name, "conv2d"))
      |> Axon.nx(fn conv -> Nx.multiply(conv, scale) end)
      |> Axon.add(x)

    if activation == :linear do
      residual
    else
      Axon.activation(residual, activation, name: join(name, "relu"))
    end
  end

  # Mixed_6a: Reduction-A (downsampling transition)
  defp mixed_6a(x, opts) do
    name = opts[:name]

    branch0 = basic_conv2d(x, 384, kernel_size: 3, strides: 2, name: join(name, "branch0"))

    branch1 =
      x
      |> basic_conv2d(192, kernel_size: 1, strides: 1, name: join(name, "branch1.0"))
      |> basic_conv2d(192, kernel_size: 3, strides: 1, padding: [{1, 1}, {1, 1}], name: join(name, "branch1.1"))
      |> basic_conv2d(256, kernel_size: 3, strides: 2, name: join(name, "branch1.2"))

    branch2 = Axon.max_pool(x, kernel_size: 3, strides: 2, name: join(name, "branch2"))

    Axon.concatenate([branch0, branch1, branch2], axis: 3)
  end

  # Mixed_7a: Reduction-B (downsampling transition)
  defp mixed_7a(x, opts) do
    name = opts[:name]

    branch0 =
      x
      |> basic_conv2d(256, kernel_size: 1, strides: 1, name: join(name, "branch0.0"))
      |> basic_conv2d(384, kernel_size: 3, strides: 2, name: join(name, "branch0.1"))

    branch1 =
      x
      |> basic_conv2d(256, kernel_size: 1, strides: 1, name: join(name, "branch1.0"))
      |> basic_conv2d(256, kernel_size: 3, strides: 2, name: join(name, "branch1.1"))

    branch2 =
      x
      |> basic_conv2d(256, kernel_size: 1, strides: 1, name: join(name, "branch2.0"))
      |> basic_conv2d(256, kernel_size: 3, strides: 1, padding: [{1, 1}, {1, 1}], name: join(name, "branch2.1"))
      |> basic_conv2d(256, kernel_size: 3, strides: 2, name: join(name, "branch2.2"))

    branch3 = Axon.max_pool(x, kernel_size: 3, strides: 2, name: join(name, "branch3"))

    Axon.concatenate([branch0, branch1, branch2, branch3], axis: 3)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          num_channels: {"num_channels", number()},
          image_size: {"image_size", number()},
          dropout_prob: {"dropout_prob", number()},
          embedding_size: {"embedding_size", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        # Stem layers
        "stem.conv2d_1a.{layer}" => "conv2d_1a.{layer}",
        "stem.conv2d_2a.{layer}" => "conv2d_2a.{layer}",
        "stem.conv2d_2b.{layer}" => "conv2d_2b.{layer}",
        "stem.conv2d_3b.{layer}" => "conv2d_3b.{layer}",
        "stem.conv2d_4a.{layer}" => "conv2d_4a.{layer}",
        "stem.conv2d_4b.{layer}" => "conv2d_4b.{layer}",
        # Block35 (repeat_1) - has branch0, branch1.0, branch1.1, branch2.0, branch2.1, branch2.2
        "blocks.repeat_1.{n}.branch0.{layer}" => "repeat_1.{n}.branch0.{layer}",
        "blocks.repeat_1.{n}.branch1.{m}.{layer}" => "repeat_1.{n}.branch1.{m}.{layer}",
        "blocks.repeat_1.{n}.branch2.{m}.{layer}" => "repeat_1.{n}.branch2.{m}.{layer}",
        "blocks.repeat_1.{n}.conv2d" => "repeat_1.{n}.conv2d",
        # Mixed 6a - has branch0, branch1.0, branch1.1, branch1.2
        "blocks.mixed_6a.branch0.{layer}" => "mixed_6a.branch0.{layer}",
        "blocks.mixed_6a.branch1.{m}.{layer}" => "mixed_6a.branch1.{m}.{layer}",
        # Block17 (repeat_2) - has branch0, branch1.0, branch1.1, branch1.2
        "blocks.repeat_2.{n}.branch0.{layer}" => "repeat_2.{n}.branch0.{layer}",
        "blocks.repeat_2.{n}.branch1.{m}.{layer}" => "repeat_2.{n}.branch1.{m}.{layer}",
        "blocks.repeat_2.{n}.conv2d" => "repeat_2.{n}.conv2d",
        # Mixed 7a - has branch0.0, branch0.1, branch1.0, branch1.1, branch2.0, branch2.1, branch2.2
        "blocks.mixed_7a.branch0.{m}.{layer}" => "mixed_7a.branch0.{m}.{layer}",
        "blocks.mixed_7a.branch1.{m}.{layer}" => "mixed_7a.branch1.{m}.{layer}",
        "blocks.mixed_7a.branch2.{m}.{layer}" => "mixed_7a.branch2.{m}.{layer}",
        # Block8 (repeat_3) - has branch0, branch1.0, branch1.1, branch1.2
        "blocks.repeat_3.{n}.branch0.{layer}" => "repeat_3.{n}.branch0.{layer}",
        "blocks.repeat_3.{n}.branch1.{m}.{layer}" => "repeat_3.{n}.branch1.{m}.{layer}",
        "blocks.repeat_3.{n}.conv2d" => "repeat_3.{n}.conv2d",
        # Final Block8 - has branch0, branch1.0, branch1.1, branch1.2
        "blocks.block8.branch0.{layer}" => "block8.branch0.{layer}",
        "blocks.block8.branch1.{m}.{layer}" => "block8.branch1.{m}.{layer}",
        "blocks.block8.conv2d" => "block8.conv2d",
        # Head
        "head.last_linear" => "last_linear",
        "head.last_bn" => "last_bn",
        # Classification head
        "image_classification_head.output" => "logits"
      }
    end
  end
end
