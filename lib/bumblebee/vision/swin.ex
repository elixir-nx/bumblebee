defmodule Bumblebee.Vision.Swin do
  alias Bumblebee.Shared

  options =
    [
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      depths: [
        default: [2, 2, 18, 2],
        doc: "the depth (number of residual blocks) at each stage"
      ],
      drop_path_rate: [
        default: 0.1,
        doc: "the drop path rate used to for stochastic depth"
      ],
      # Maybe it should be renamed to hidden_size
      embed_dim: [
        default: 128,
        doc: ""
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder and decoder"
      ],
      image_size: [
        default: 384,
        doc: "the size of the input spatial dimensions"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      intermediate_size_ratio: [
        default: 4,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder,
        expressed as a multiplier of `:hidden_size`
        """
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      num_heads: [
        default: [4, 8, 16, 32],
        doc: "number of attention heads"
      ],
      patch_size: [
        default: 4,
        doc: "the size of the patch spatial dimensions"
      ],
      path_norm: [
        default: true,
        doc: ""
      ],
      use_attention_bias: [
        default: true,
        doc: "whether to use bias in query, key, and value projections"
      ],
      use_absolute_embeddings: [
        default: false,
        doc: ""
      ],
      window_size: [
        default: 12,
        doc: ""
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  Swin Transformer model.

  ## Architectures

    * `:for_image_classification` - Swin tranformer model for image classification.

  ## Global layer options

  # {Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  # {Shared.options_doc(options)}

  ## References

    * [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:for_image_classification]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(spec) do
    %{
      "pixel_values" =>
        Nx.template({1, spec.image_size, spec.image_size, spec.num_channels}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    spec
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Layers.take_token(index: 0, axis: 1)
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "image_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape),
      Axon.input("patch_mask", shape: {nil, nil}, optional: true)
    ])
  end

  # Contrary to Python implementation we do not have here argument
  # bool_maked_pos. This parameter is propagated from model through
  # core to embedder.
  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["pixel_values"], spec, name: join(name, "embedder"))

    {hidden_state, hidden_states, attentions} =
      encoder(embeddings, spec, name: join(name, "encoder"))

    hidden_state =
      Axon.layer_norm(hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "norm")
      )

    pooled_state =
      Axon.adaptive_avg_pool(hidden_state, output_size: {1, 1}, name: join(name, "pooler"))

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: hidden_states,
      attentions: attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    embeddings =
      pixel_values
      |> patch_embedding(spec, name: join(name, "patch_embedding"))
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon)

    embeddings =
      if spec.use_absolute_embeddings do
        num_patches = div(spec.image_size, spec.patch_size) ** 2

        position_embeddings =
          Layers.learned_embeddings(num_patches, spec.embed_dim,
            initializer: :zeros,
            name: join(name, "position_embedding")
          )

        Axon.add(embeddings, position_embeddings)
      else
        embeddings
      end

    embeddings
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  # TODO: How to get and return output dimensions
  # They are used in Python later but here it is not clear
  # how to get them till we have loadable implementation.
  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]
    hidden_size = spec.embed_dim

    pixel_values
    |> Axon.conv(hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection")
    )
    |> Axon.reshape({:batch, :auto, spec.embed_dim}, name: join(name, "reshape"))
  end

  defp encoder(hidden_state, spec, opts) do
    hidden_states = Axon.container({hidden_state})
    attentions = Axon.container({})

    0..(length(spec.depths) - 1)
    |> Enum.reduce(
      {hidden_state, hidden_states, attentions},
      fn layer_idx, {hidden_state, hidden_states, attentions} ->
        {hidden_state, attention, _cross_attention, _block_cache, _position_bias} =
          stage(hidden_state, spec, layer_idx, opts)

        {
          hidden_state,
          Layers.append(hidden_states, hidden_state),
          Layers.append(attentions, attention)
        }
      end
    )
  end

  defp stage(hidden_state, spec, layer_idx, opts) do
    grid_size = div(spec.image_size, spec.patch_size)
    input_resolution = div(grid_size, 2 ** layer_idx)
    num_attention_heads = Enum.at(spec.num_heads, layer_idx)
    dim = spec.embed_dim * 2 ** layer_idx

    {hidden_state, attention, cross_attention, block_cache, position_bias} =
      layer(hidden_state, num_attention_heads, dim, spec, opts)

    hidden_state =
      if layer_idx < length(spec.depths) - 1 do
        downsample(hidden_state, input_resolution, dim, spec.layer_norm_epsilon)
      else
        hidden_state
      end

    {hidden_state, attention, cross_attention, block_cache, position_bias}
  end

  # Steps in Python implementation:
  # Normalization
  # if shift_size > 0 -> roll hidden states
  # window partition
  # attention with attention mask
  # window reverse
  # if shift_size > 0 -> roll shifted windows
  # shortcut + drop_path(attention_windows)
  # Normalization
  # Intermediate
  # add result of intermediate
  defp layer(hidden_state, num_attention_heads, dim, spec, opts) do
    name = opts[:name]

    # shift_size = if 0 == rem(layer_idx, 2), do: 0, else: div(spec.window_size, 2)
    # depth = Enum.at(spec.depths, layer_idx)

    {hidden_state, attention, cross_attention, block_cache, position_bias} =
      Layers.Transformer.block(hidden_state,
        block_type: :norm_first,
        num_attention_heads: num_attention_heads,
        hidden_size: dim,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [
          epsilon: spec.layer_norm_epsilon
        ],
        ffn: [
          intermediate_size: round(spec.intermediate_size_ratio * dim),
          activation: spec.activation
        ],
        name: join(name, "block_#{num_attention_heads}")
      )

    {hidden_state, attention, cross_attention, block_cache, position_bias}
  end

  defp downsample(hidden_state, input_resolution, dim, norm_epsilon) do
    Axon.nx(hidden_state, fn x ->
      {batch_size, _dim, num_channels} = Nx.shape(x)

      x = Nx.reshape(x, {batch_size, input_resolution, input_resolution, :auto})

      input_feature_0 = x[[.., 0..-1//2, 0..-1//2, ..]]
      input_feature_1 = x[[.., 1..-1//2, 0..-1//2, ..]]
      input_feature_2 = x[[.., 0..-1//2, 1..-1//2, ..]]
      input_feature_3 = x[[.., 1..-1//2, 1..-1//2, ..]]

      Nx.concatenate([input_feature_0, input_feature_1, input_feature_2, input_feature_3],
        axis: -1
      )
      |> Nx.reshape({batch_size, :auto, 4 * num_channels})
    end)
    |> Axon.layer_norm(epsilon: norm_epsilon, name: "downsample_norm")
    |> Axon.dense(2 * dim,
      kernel_initializer: Axon.Initializers.uniform(),
      name: "image_classification_head.output"
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          depths: {"depths", list(number())},
          drop_path_rate: {"drop_path_rate", number()},
          embed_dim: {"embed_dim", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          image_size: {"image_size", number()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          intermediate_size_ratio: {"mlp_ratio", number()},
          num_channels: {"num_channels", number()},
          num_heads: {"num_heads", list(number())},
          patch_size: {"patch_size", number()},
          path_norm: {"path_norm", boolean()},
          use_attention_bias: {"qkv_bias", boolean()},
          use_absolute_embeddings: {"use_absolute_embeddings", boolean()},
          window_size: {"window_size", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding.projection" => "swin.embeddings.patch_embeddings.projection",
        "embedder.class_embedding" => %{
          "embeddings" => {
            [{"swin.embeddings", "cls_token"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "embedder.position_embedding" => %{
          "embeddings" => {
            [{"swin.embeddings", "position_embeddings"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "encoder.block_{n}.self_attention_norm" => "swin.encoder.layer.{n}.layernorm_before",
        "encoder.block_{n}.self_attention.key" =>
          "swin.encoder.layer.{n}.attention.attention.key",
        "encoder.block_{n}.self_attention.query" =>
          "swin.encoder.layer.{n}.attention.attention.query",
        "encoder.block_{n}.self_attention.value" =>
          "swin.encoder.layer.{n}.attention.attention.value",
        "encoder.block_{n}.self_attention.output" =>
          "swin.encoder.layer.{n}.attention.output.dense",
        "encoder.block_{n}.ffn.intermediate" => "swin.encoder.layer.{n}.intermediate.dense",
        "encoder.block_{n}.ffn.output" => "swin.encoder.layer.{n}.output.dense",
        "encoder.block_{n}.output_norm" => "swin.encoder.layer.{n}.layernorm_after",
        "norm" => "swin.layernorm",
        "pooler.output" => "swin.pooler.dense",
        "image_classification_head.output" => "classifier",
        "masked_image_modeling_head.output" => "decoder.0",
        "layer_norm_{n}" => "swin.encoder.layers.{n}.blocks.{n}.layernorm",
        "layer_{n}_downsample_norm" => "swin.encoder.layers.{n}.downsample.norm",
        "downsample_norm" => "swin.encoder.downsample.norm"
      }
    end
  end

  defp roll(%Nx.Tensor{} = x, opts \\ []) do
    opts = Keyword.validate!(opts, shifts: [], axes: [])
    shifts = opts[:shifts]
    axes = opts[:axes]

    if length(shifts) != length(axes) do
      raise ArgumentError, "shifts and axes must align, shifts: #{shifts}, axes: #{axes}"
    else
      shape = Nx.shape(x) |> Tuple.to_list()

      Enum.zip(shifts, axes)
      |> Enum.reduce(x, fn {shift, dim}, acc ->
        shift = rem(shift, Enum.at(shape, dim)) |> IO.inspect(label: :shift)

        if 0 < shift do
          {base, move} = Nx.split(acc, -1 * shift, axis: dim) |> IO.inspect()
          Nx.concatenate([move, base], axis: dim)
        else
          acc
        end
      end)
    end
  end
end
