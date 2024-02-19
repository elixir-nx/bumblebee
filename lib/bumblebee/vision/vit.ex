defmodule Bumblebee.Vision.Vit do
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
        default: 16,
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
      use_qkv_bias: [
        default: true,
        doc: "whether to use bias in query, key, and value projections"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
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
        default: 1.0e-12,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ])

  @moduledoc """
  ViT model family.

  ## Architectures

    * `:base` - plain ViT without any head on top

    * `:for_image_classification` - ViT with a classification head.
      The head consists of a single dense layer on top of the pooled
      features

    * `:for_masked_image_modeling` - ViT with a language modeling
      head on top for predicting visual tokens

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

    * `"patch_mask"` - `{batch_size, num_patches}`

      Mask to nullify selected embedded patches.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_image_classification, :for_masked_image_modeling]

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

  def model(%__MODULE__{architecture: :for_masked_image_modeling} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    pixel_values =
      outputs.hidden_state
      |> Axon.nx(fn x ->
        x = x[[.., 1..-1//1]]
        {batch_size, sequence_length, channels} = Nx.shape(x)
        height = width = sequence_length |> :math.sqrt() |> floor()
        Nx.reshape(x, {batch_size, height, width, channels})
      end)
      # Upsample to the original spatial resolution
      |> Axon.conv(spec.patch_size ** 2 * 3,
        kernel_size: 1,
        kernel_initializer: kernel_initializer(spec),
        name: "masked_image_modeling_head.output"
      )
      |> Layers.pixel_shuffle(spec.patch_size)

    Layers.output(%{
      pixel_values: pixel_values,
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

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["pixel_values"], inputs["patch_mask"], spec, name: join(name, "embedder"))

    encoder_outputs = encoder(embeddings, spec, name: join(name, "encoder"))

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "norm")
      )

    pooled_state = pooler(hidden_state, spec, name: join(name, "pooler"))

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
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

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

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

  defp pooler(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(index: 0, axis: 1)
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.tanh()
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
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          use_qkv_bias: {"qkv_bias", boolean()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding.projection" => "vit.embeddings.patch_embeddings.projection",
        "embedder.class_embedding" => %{
          "embeddings" => {
            [{"vit.embeddings", "cls_token"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "embedder.position_embedding" => %{
          "embeddings" => {
            [{"vit.embeddings", "position_embeddings"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "encoder.blocks.{n}.self_attention_norm" => "vit.encoder.layer.{n}.layernorm_before",
        "encoder.blocks.{n}.self_attention.key" =>
          "vit.encoder.layer.{n}.attention.attention.key",
        "encoder.blocks.{n}.self_attention.query" =>
          "vit.encoder.layer.{n}.attention.attention.query",
        "encoder.blocks.{n}.self_attention.value" =>
          "vit.encoder.layer.{n}.attention.attention.value",
        "encoder.blocks.{n}.self_attention.output" =>
          "vit.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.ffn.intermediate" => "vit.encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => "vit.encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => "vit.encoder.layer.{n}.layernorm_after",
        "norm" => "vit.layernorm",
        "pooler.output" => "vit.pooler.dense",
        "image_classification_head.output" => "classifier",
        "masked_image_modeling_head.output" => "decoder.0"
      }
    end
  end
end
