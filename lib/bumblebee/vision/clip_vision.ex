defmodule Bumblebee.Vision.ClipVision do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 224,
        doc: "the size of the input spatial dimensions"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 32,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 768,
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
      intermediate_size: [
        default: 3072,
        docs:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      projection_size: [
        default: 512,
        doc: "the dimensionality of the projection layer"
      ],
      activation: [
        default: :gelu_approx_sigmoid,
        doc: "the activation function"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions
      ])

  @moduledoc """
  The CLIP model for image encoding.

  ## Architectures

    * `:base` - the base image model

    * `:for_embedding` - the base model with a single projection layer
      on top. The head returns a vector embedded in the joint text-image
      CLIP space

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_embedding]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
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

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_embedding} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    embedding =
      outputs.pooled_state
      |> Axon.dense(spec.projection_size, use_bias: false, name: "projection_head.output")

    Layers.output(%{
      embedding: embedding,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp core(inputs, spec) do
    embeddings = embedder(inputs["pixel_values"], spec, name: "embedder")

    encoder_outputs =
      embeddings
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: "pre_norm")
      |> encoder(spec, name: "encoder")

    pooled_state =
      encoder_outputs.hidden_state
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: "post_norm")
      |> Layers.take_token(index: 0, axis: 1)

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    patch_embeddings = patch_embedding(pixel_values, spec, name: join(name, "patch_embedding"))

    class_embedding =
      Layers.learned_embeddings(1, spec.hidden_size,
        name: join(name, "class_embedding"),
        initializer: Axon.Initializers.normal()
      )

    input_embeddings = Layers.concatenate_embeddings([class_embedding, patch_embeddings])

    num_patches = div(spec.image_size, spec.patch_size) ** 2
    num_positions = num_patches + 1
    position_ids = position_ids(num_positions)

    position_embeddings =
      Axon.embedding(position_ids, num_patches + 1, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    Axon.add(input_embeddings, position_embeddings)
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: Axon.Initializers.normal(),
      use_bias: false,
      name: name
    )
    |> Axon.reshape({:batch, :auto, spec.hidden_size}, name: join(name, "reshape"))
  end

  defp position_ids(num_position_ids) do
    Axon.layer(
      fn _opts -> Nx.iota({1, num_position_ids}) end,
      [],
      op_name: :position_ids
    )
  end

  defp encoder(embeddings, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(embeddings,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: Axon.Initializers.normal(scale: 0.01),
      dropout_rate: 0.0,
      attention_dropout_rate: spec.attention_dropout_rate,
      layer_norm: [
        epsilon: spec.layer_norm_epsilon
      ],
      ffn: [
        intermediate_size: spec.intermediate_size,
        activation: spec.activation
      ],
      block_type: :norm_first,
      output_hidden_states: spec.output_hidden_states,
      output_attentions: spec.output_attentions,
      name: join(name, "blocks")
    )
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    # Support loading from the entire Clip configuration
    def load(spec, %{"model_type" => "clip", "vision_config" => data}) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          image_size: {"image_size", number()},
          patch_size: {"patch_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          projection_size: {"projection_dim", number()},
          activation: {"hidden_act", activation()},
          attention_dropout_rate: {"attention_dropout", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding" => "vision_model.embeddings.patch_embedding",
        "embedder.class_embedding" => %{
          "embeddings" => {
            [{"vision_model.embeddings", "class_embedding"}],
            fn [value] -> Nx.new_axis(value, 0) end
          }
        },
        "embedder.position_embedding" => "vision_model.embeddings.position_embedding",
        "encoder.blocks.{n}.self_attention_norm" => "vision_model.encoder.layers.{n}.layer_norm1",
        "encoder.blocks.{n}.self_attention.query" =>
          "vision_model.encoder.layers.{n}.self_attn.q_proj",
        "encoder.blocks.{n}.self_attention.key" =>
          "vision_model.encoder.layers.{n}.self_attn.k_proj",
        "encoder.blocks.{n}.self_attention.value" =>
          "vision_model.encoder.layers.{n}.self_attn.v_proj",
        "encoder.blocks.{n}.self_attention.output" =>
          "vision_model.encoder.layers.{n}.self_attn.out_proj",
        "encoder.blocks.{n}.ffn.intermediate" => "vision_model.encoder.layers.{n}.mlp.fc1",
        "encoder.blocks.{n}.ffn.output" => "vision_model.encoder.layers.{n}.mlp.fc2",
        "encoder.blocks.{n}.output_norm" => "vision_model.encoder.layers.{n}.layer_norm2",
        "pre_norm" => "vision_model.pre_layrnorm",
        "post_norm" => "vision_model.post_layernorm",
        "projection_head.output" => "visual_projection"
      }
    end
  end
end
