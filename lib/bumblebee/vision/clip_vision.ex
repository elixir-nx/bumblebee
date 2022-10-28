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
          "the dimensionality of the intermediate (often named feed-forward) layer in the encoder"
      ],
      activation: [
        default: :quick_gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder"
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
        :output_attentions,
        :num_labels,
        :id_to_label
      ])

  @moduledoc """
  The CLIP model for image encoding.

  ## Architectures

    * `:base` - the base image model

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
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts \\ []) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(spec) do
    %{
      "pixel_values" =>
        Nx.template({1, spec.image_size, spec.image_size, spec.num_channels}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> clip_vision(spec, name: "vision_model")
    |> Layers.output()
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp clip_vision(inputs, spec, opts) do
    name = opts[:name]

    embeddings =
      inputs
      |> embeddings(spec, name: join(name, "embeddings"))
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "pre_layrnorm"))

    attention_mask = Layers.default_attention_mask(embeddings)

    encoder_outputs =
      Bumblebee.Layers.Clip.encoder(embeddings, attention_mask, spec, name: join(name, "encoder"))

    pooled_state =
      encoder_outputs.hidden_state
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "post_layernorm"))
      |> Layers.take_token(index: 0, axis: 1, name: join(name, "head"))

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embeddings(inputs, spec, opts) do
    name = opts[:name]

    pixel_values = inputs["pixel_values"]

    patch_embeddings = patch_embeddings(pixel_values, spec, name: join(name, "patch_embedding"))

    num_patches = div(spec.image_size, spec.patch_size) ** 2

    class_embeddings =
      Axon.param("class_embedding", fn _, _ -> {spec.hidden_size} end,
        initializer: Axon.Initializers.normal()
      )

    num_positions = num_patches + 1
    position_ids = position_ids(num_positions)

    position_embeddings =
      Axon.embedding(position_ids, num_positions, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    Axon.layer(
      fn patch_embeddings, class_embeddings, position_embeddings, _opts ->
        batch_size = Nx.axis_size(patch_embeddings, 0)

        class_embeddings =
          class_embeddings
          |> Nx.reshape({1, 1, :auto})
          |> Nx.broadcast({batch_size, 1, spec.hidden_size})

        Nx.concatenate([class_embeddings, patch_embeddings], axis: 1)
        |> Nx.add(position_embeddings)
      end,
      [patch_embeddings, class_embeddings, position_embeddings],
      name: name
    )
  end

  defp patch_embeddings(pixel_values, spec, opts) do
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
          activation: {"hidden_act", atom()},
          dropout_rate: {"dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
