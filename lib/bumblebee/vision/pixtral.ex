defmodule Bumblebee.Vision.Pixtral do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 1540,
        doc: "the size of the input spatial dimensions"
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
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      head_dim: [
        default: 64,
        doc: "the dimensionality of each attention head"
      ],
      intermediate_size: [
        default: 4096,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      rotary_embedding_base: [
        default: 10_000.0,
        doc: "base for computing rotary embedding frequency"
      ]
    ]

  @moduledoc """
  Pixtral vision encoder model.

  Pixtral is a Vision Transformer variant used in Mistral3 multimodal models.
  It uses Rotary Position Embeddings (RoPE) instead of learned position embeddings.

  ## Architectures

    * `:base` - plain Pixtral encoder without any head on top

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

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
    spec
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings = embedder(inputs["pixel_values"], spec, name: join(name, "embedder"))

    # Position IDs for RoPE - use default 1D positions (flattened from 2D grid)
    position_ids = Layers.default_position_ids(embeddings)

    encoder_outputs =
      encoder(embeddings, position_ids, spec, name: join(name, "encoder"))

    hidden_state =
      Layers.rms_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "norm")
      )

    %{
      hidden_state: hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    # Patch embedding without class token (Pixtral doesn't use CLS token)
    # Note: Pixtral patch_conv does not use bias
    pixel_values
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      use_bias: false,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "patch_embedding.projection")
    )
    |> Axon.reshape({:batch, :auto, spec.hidden_size}, name: join(name, "reshape"))
  end

  defp encoder(hidden_state, position_ids, spec, opts) do
    name = opts[:name]

    num_patches = div(spec.image_size, spec.patch_size) ** 2

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.head_dim,
      kernel_initializer: kernel_initializer(spec),
      attention_dropout_rate: spec.attention_dropout_rate,
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      output_use_bias: false,
      layer_norm: &Layers.rms_norm(&1, name: &2, epsilon: spec.layer_norm_epsilon),
      ffn:
        &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
          name: &2,
          activation: spec.activation
        ),
      rotary_embedding: [
        position_ids: position_ids,
        max_positions: num_patches,
        base: spec.rotary_embedding_base
      ],
      block_type: :norm_first,
      name: join(name, "blocks")
    )
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    gate = Axon.dense(hidden_state, intermediate_size, name: join(name, "gate"), use_bias: false)

    hidden_state = Axon.multiply(intermediate, Layers.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
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
          head_dim: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          attention_dropout_rate: {"attention_dropout", number()},
          layer_norm_epsilon: {"rms_norm_eps", optional(number())},
          initializer_scale: {"initializer_range", number()},
          rotary_embedding_base: {"rope_theta", number()}
        )

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding.projection" => "patch_conv",
        "encoder.blocks.{n}.self_attention.query" => "transformer.layers.{n}.attention.q_proj",
        "encoder.blocks.{n}.self_attention.key" => "transformer.layers.{n}.attention.k_proj",
        "encoder.blocks.{n}.self_attention.value" => "transformer.layers.{n}.attention.v_proj",
        "encoder.blocks.{n}.self_attention.output" => "transformer.layers.{n}.attention.o_proj",
        "encoder.blocks.{n}.self_attention_norm" => "transformer.layers.{n}.attention_norm",
        "encoder.blocks.{n}.ffn.gate" => "transformer.layers.{n}.feed_forward.gate_proj",
        "encoder.blocks.{n}.ffn.intermediate" => "transformer.layers.{n}.feed_forward.up_proj",
        "encoder.blocks.{n}.ffn.output" => "transformer.layers.{n}.feed_forward.down_proj",
        "encoder.blocks.{n}.output_norm" => "transformer.layers.{n}.ffn_norm",
        "norm" => "ln_pre"
      }
    end
  end
end
