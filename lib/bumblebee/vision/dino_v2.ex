defmodule Bumblebee.Vision.DinoV2 do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 518,
        doc: """
        the size of the input spatial dimensions. The model is trained for this size, however
        the model supports any other input size by interpolating position embeddings
        """
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 14,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 384,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size_ratio: [
        default: 4,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder,
        expressed as a multiplier of `:hidden_size`
        """
      ],
      use_qkv_bias: [
        default: true,
        doc: "whether to use bias in query, key, and value projections"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      ffn_swiglu_activation: [
        default: false,
        doc:
          "whether to use the gated SwiGLU activation function in the feed-forward network (FFN)"
      ],
      scale_initial_value: [
        default: 1.0,
        doc: "the initial value for scaling layers"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      backbone_output_indices: [
        default: nil,
        doc: """
        list of indices indicating which feature maps to include in the output. If not specified, only
        the last feature map is included
        """
      ],
      backbone_use_norm: [
        default: true,
        doc:
          "whether to add layer normalization layer to each of the feature maps returned by the backbone"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ])

  @moduledoc """
  DINOv2 model family.

  ## Architectures

    * `:base` - plain DINOv2 without any head on top

    * `:for_image_classification` - DINOv2 with head for image classification

    * `:backbone` - DINOv2 with feature maps output

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

    * `"patch_mask"` - `{batch_size, num_patches}`

      Mask to nullify selected embedded patches.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :backbone, :for_image_classification]

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
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    hidden_state =
      Axon.layer_norm(outputs.hidden_state, epsilon: spec.layer_norm_epsilon, name: "norm")

    pooled_state = Layers.take_token(hidden_state, index: 0, axis: 1)

    Layers.output(%{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    hidden_state =
      Axon.layer_norm(outputs.hidden_state, epsilon: spec.layer_norm_epsilon, name: "norm")

    class_token = Layers.take_token(hidden_state, index: 0, axis: 1)

    patch_embeddings_mean =
      Axon.nx(hidden_state, fn hidden_state ->
        patch_embeddings = hidden_state[[.., 1..-1//1, ..]]
        Nx.mean(patch_embeddings, axes: [1])
      end)

    logits =
      Axon.concatenate(class_token, patch_embeddings_mean)
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

  def model(%__MODULE__{architecture: :backbone} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, %{spec | output_hidden_states: true})
    feature_maps = feature_maps(outputs.hidden_states, inputs["pixel_values"], spec)

    Layers.output(%{
      feature_maps: feature_maps,
      hidden_states:
        if(spec.output_hidden_states, do: outputs.hidden_states, else: Layers.none()),
      attentions: outputs.attentions
    })
  end

  defp inputs(spec) do
    shape = {nil, nil, nil, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape),
      Axon.input("patch_mask", shape: {nil, nil}, optional: true)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["pixel_values"], inputs["patch_mask"], spec, name: join(name, "embedder"))

    encoder_outputs = encoder(embeddings, spec, name: join(name, "encoder"))

    %{
      hidden_state: encoder_outputs.hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp feature_maps(hidden_states, pixel_values, spec, opts \\ []) do
    name = opts[:name]

    num_feature_maps = spec.num_blocks + 1
    output_indices = spec.backbone_output_indices || [num_feature_maps - 1]

    for index <- output_indices do
      hidden_state = Axon.nx(hidden_states, &elem(&1, index))

      hidden_state =
        if spec.backbone_use_norm do
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(name, "norm")
          )
        else
          hidden_state
        end

      Axon.layer(
        fn hidden_state, pixel_values, _opts ->
          {batch_size, height, width, _channels} = Nx.shape(pixel_values)

          hidden_state = hidden_state[[.., 1..-1//1, ..]]

          Nx.reshape(
            hidden_state,
            {batch_size, div(height, spec.patch_size), div(width, spec.patch_size), :auto}
          )
        end,
        [hidden_state, pixel_values]
      )
    end
    |> List.to_tuple()
    |> Axon.container()
  end

  defp embedder(pixel_values, patch_mask, spec, opts) do
    name = opts[:name]

    patch_embeddings =
      pixel_values
      |> patch_embedding(spec, name: join(name, "patch_embedding"))
      |> Layers.apply_vision_patch_mask(patch_mask, name: join(name, "mask_tokens"))

    class_embedding =
      Layers.learned_embeddings(1, spec.hidden_size, name: join(name, "class_embedding"))

    input_embeddings = Layers.concatenate_embeddings([class_embedding, patch_embeddings])

    num_patches = div(spec.image_size, spec.patch_size) ** 2

    position_embeddings =
      Layers.learned_embeddings(num_patches + 1, spec.hidden_size,
        initializer: :zeros,
        name: join(name, "position_embedding")
      )
      |> interpolate_position_embeddings(pixel_values, spec)

    Axon.add(input_embeddings, position_embeddings)
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection")
    )
    |> Axon.reshape({:batch, :auto, spec.hidden_size}, name: join(name, "reshape"))
  end

  defp interpolate_position_embeddings(position_embeddings, pixel_values, spec) do
    Axon.layer(
      fn position_embeddings, pixel_values, _opts ->
        original_positions = div(spec.image_size, spec.patch_size)
        {batch_size, height, width, _channels} = Nx.shape(pixel_values)
        resized_height = div(height, spec.patch_size)
        resized_width = div(width, spec.patch_size)

        class_position_embedding = position_embeddings[[.., 0..0//1, ..]]
        input_position_embeddings = position_embeddings[[.., 1..-1//1, ..]]

        interpolated_position_embeddings =
          input_position_embeddings
          |> Nx.reshape({batch_size, original_positions, original_positions, spec.hidden_size})
          |> Axon.Layers.resize(
            size: {resized_height, resized_width},
            method: :bicubic,
            antialias: false
          )
          |> Nx.reshape({batch_size, :auto, spec.hidden_size})

        Nx.concatenate([class_position_embedding, interpolated_position_embeddings], axis: 1)
      end,
      [position_embeddings, pixel_values],
      op_name: :interpolate_position_embeddings
    )
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    ffn =
      if spec.ffn_swiglu_activation do
        intermediate_size =
          div(floor(floor(spec.hidden_size * spec.intermediate_size_ratio) * 2 / 3 + 7), 8) * 8

        &ffn_swiglu(&1, intermediate_size, spec.hidden_size, name: &2)
      else
        intermediate_size = floor(spec.hidden_size * spec.intermediate_size_ratio)

        [
          intermediate_size: intermediate_size,
          activation: spec.activation
        ]
      end

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: spec.dropout_rate,
      attention_dropout_rate: spec.attention_dropout_rate,
      query_use_bias: spec.use_qkv_bias,
      key_use_bias: spec.use_qkv_bias,
      value_use_bias: spec.use_qkv_bias,
      layer_norm: [
        epsilon: spec.layer_norm_epsilon
      ],
      ffn: ffn,
      block_type: &block_impl(&1, &2, &3, spec),
      output_hidden_states: spec.output_hidden_states,
      output_attentions: spec.output_attentions,
      name: join(name, "blocks")
    )
  end

  # A feed-forward network with SwiGLU nonlinearity as in https://arxiv.org/abs/2002.05202
  defp ffn_swiglu(x, intermediate_size, output_size, opts) do
    name = opts[:name]
    dropout = opts[:dropout] || 0.0

    {gate, x} =
      x
      |> Axon.dense(intermediate_size * 2, name: join(name, "intermediate"))
      |> Axon.split(2, axis: -1)

    x = Axon.multiply(x, Axon.silu(gate))

    x
    |> Axon.dropout(rate: dropout, name: join(name, "dropout"))
    |> Axon.dense(output_size, name: join(name, "output"))
  end

  # :norm_first block with additional scaling layers
  defp block_impl(hidden_state, steps, name, spec) do
    shortcut = hidden_state

    {hidden_state, attention_info} =
      hidden_state
      |> steps.self_attention_norm.()
      |> steps.self_attention.()

    hidden_state =
      hidden_state
      |> Bumblebee.Layers.scale(
        scale_initializer: Axon.Initializers.full(spec.scale_initial_value),
        name: join(name, "self_attention_scale")
      )
      |> Axon.add(shortcut)

    {_hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn _hidden_state ->
        raise "cross attention not supported"
      end)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> steps.output_norm.()
      |> steps.ffn.()
      |> Bumblebee.Layers.scale(
        scale_initializer: Axon.Initializers.full(spec.scale_initial_value),
        name: join(name, "output_scale")
      )
      |> Axon.add(shortcut)

    {hidden_state, attention_info, cross_attention_info}
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          image_size: {"image_size", number()},
          num_channels: {"num_channels", number()},
          patch_size: {"patch_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size_ratio: {"mlp_ratio", number()},
          activation: {"hidden_act", activation()},
          use_qkv_bias: {"qkv_bias", boolean()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          scale_initial_value: {"layerscale_value", number()},
          ffn_swiglu_activation: {"use_swiglu_ffn", boolean()},
          backbone_output_indices: {"out_indices", list(number())},
          backbone_use_norm: {"use_backbone_norm", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.patch_embedding.projection" => "dinov2.embeddings.patch_embeddings.projection",
        "embedder.class_embedding" => %{
          "embeddings" => {
            [{"dinov2.embeddings", "cls_token"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "embedder.position_embedding" => %{
          "embeddings" => {
            [{"dinov2.embeddings", "position_embeddings"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "encoder.blocks.{n}.self_attention_norm" => "dinov2.encoder.layer.{n}.norm1",
        "encoder.blocks.{n}.self_attention.key" =>
          "dinov2.encoder.layer.{n}.attention.attention.key",
        "encoder.blocks.{n}.self_attention.query" =>
          "dinov2.encoder.layer.{n}.attention.attention.query",
        "encoder.blocks.{n}.self_attention.value" =>
          "dinov2.encoder.layer.{n}.attention.attention.value",
        "encoder.blocks.{n}.self_attention.output" =>
          "dinov2.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.self_attention_scale" => %{
          "scale" => {
            [{"dinov2.encoder.layer.{n}.layer_scale1", "lambda1"}],
            fn [lambda1] -> lambda1 end
          }
        },
        "encoder.blocks.{n}.ffn.intermediate" =>
          if(spec.ffn_swiglu_activation,
            do: "dinov2.encoder.layer.{n}.mlp.weights_in",
            else: "dinov2.encoder.layer.{n}.mlp.fc1"
          ),
        "encoder.blocks.{n}.ffn.output" =>
          if(spec.ffn_swiglu_activation,
            do: "dinov2.encoder.layer.{n}.mlp.weights_out",
            else: "dinov2.encoder.layer.{n}.mlp.fc2"
          ),
        "encoder.blocks.{n}.output_norm" => "dinov2.encoder.layer.{n}.norm2",
        "encoder.blocks.{n}.output_scale" => %{
          "scale" => {
            [{"dinov2.encoder.layer.{n}.layer_scale2", "lambda1"}],
            fn [lambda1] -> lambda1 end
          }
        },
        "norm" => "dinov2.layernorm",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end
