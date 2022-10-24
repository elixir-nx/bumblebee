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
          "the dimensionality of the intermediate (often named feed-forward) layer in the encoder"
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

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures(), do: [:base, :for_image_classification, :for_masked_image_modeling]

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
        Nx.template({1, spec.image_size, spec.image_size, spec.num_channels}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    outputs =
      spec
      |> inputs()
      |> vit(spec, name: "vit")

    logits =
      outputs.hidden_state
      |> Layers.take_token(index: 0, axis: 1, name: join("vit", "head"))
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "classifier"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_masked_image_modeling} = spec) do
    outputs =
      spec
      |> inputs()
      |> vit(spec, name: "vit")

    logits =
      outputs.hidden_state
      |> Axon.nx(fn x ->
        x = x[[0..-1//1, 1..-1//1]]
        {batch_size, seq_length, channels} = Nx.shape(x)
        height = width = seq_length |> :math.sqrt() |> floor()
        Nx.reshape(x, {batch_size, height, width, channels})
      end)
      # Upsample to the original spatial resolution
      |> Axon.conv(spec.patch_size ** 2 * 3,
        kernel_size: 1,
        kernel_initializer: kernel_initializer(spec),
        name: join("decoder", 0)
      )
      |> Layers.pixel_shuffle(spec.patch_size, name: join("decoder", 1))

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :base} = spec) do
    spec
    |> inputs()
    |> vit(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape),
      Axon.input("patch_mask", shape: {nil, nil}, optional: true)
    ])
  end

  defp vit(inputs, spec, opts \\ []) do
    name = opts[:name]

    hidden_state = embeddings(inputs, spec, name: join(name, "embeddings"))

    {hidden_state, hidden_states, attentions} =
      encoder(hidden_state, spec, name: join(name, "encoder"))

    hidden_state =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "layernorm")
      )

    pooled = pooler(hidden_state, spec, name: join(name, "pooler"))

    %{
      hidden_state: hidden_state,
      pooler_output: pooled,
      hidden_states: hidden_states,
      attentions: attentions
    }
  end

  defp embeddings(inputs, spec, opts) do
    name = opts[:name]

    inputs["pixel_values"]
    |> patch_embeddings(spec, name: join(name, "patch_embeddings"))
    |> Layers.apply_vision_patch_mask(inputs["patch_mask"], name: join(name, "mask_tokens"))
    |> position_embeddings(spec, name: name)
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp patch_embeddings(pixel_values, spec, opts) do
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

  defp position_embeddings(embeddings, spec, opts) do
    name = opts[:name]

    num_patches = div(spec.image_size, spec.patch_size) ** 2

    cls_token = Axon.param("cls_token", fn _ -> {1, 1, spec.hidden_size} end, initializer: :zeros)

    position_embeddings =
      Axon.param("position_embeddings", fn _ -> {1, num_patches + 1, spec.hidden_size} end,
        initializer: :zeros
      )

    Axon.layer(
      fn embeddings, cls_token, position_embeddings, _opts ->
        batch_size = Nx.axis_size(embeddings, 0)
        cls_token = Nx.broadcast(cls_token, {batch_size, 1, spec.hidden_size})

        Nx.concatenate([cls_token, embeddings], axis: 1)
        |> Nx.add(position_embeddings)
      end,
      [embeddings, cls_token, position_embeddings],
      name: name
    )
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    encoder_blocks(hidden_state, spec, name: join(name, "layer"))
  end

  defp encoder_blocks(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_states = Layers.maybe_container({hidden_state}, spec.output_hidden_states)
    attentions = Layers.maybe_container({}, spec.output_attentions)

    for idx <- 0..(spec.num_blocks - 1),
        reduce: {hidden_state, hidden_states, attentions} do
      {hidden_state, hidden_states, attentions} ->
        {hidden_state, attention} = encoder_block(hidden_state, spec, name: join(name, idx))

        {
          hidden_state,
          Layers.append(hidden_states, hidden_state),
          Layers.append(attentions, attention)
        }
    end
  end

  defp encoder_block(hidden_state, spec, opts) do
    name = opts[:name]

    {attention_output, attention} =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "layernorm_before")
      )
      |> attention(spec, name: join(name, "attention"))

    attention_output = Axon.add(attention_output, hidden_state)

    output =
      attention_output
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "layernorm_after")
      )
      |> intermediate(spec, name: join(name, "intermediate"))
      |> output(attention_output, spec, name: join(name, "output"))

    {output, attention}
  end

  defp attention(hidden_state, spec, opts) do
    name = opts[:name]

    {attention_output, attention} =
      self_attention(hidden_state, spec, name: join(name, "attention"))

    attention_output =
      self_output(attention_output, hidden_state, spec, name: join(name, "output"))

    {attention_output, attention}
  end

  defp self_attention(hidden_state, spec, opts) do
    name = opts[:name]

    num_heads = spec.num_attention_heads

    query =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        use_bias: spec.use_qkv_bias,
        name: join(name, "query")
      )
      |> Layers.split_heads(num_heads)

    key =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        use_bias: spec.use_qkv_bias,
        name: join(name, "key")
      )
      |> Layers.split_heads(num_heads)

    value =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        use_bias: spec.use_qkv_bias,
        name: join(name, "value")
      )
      |> Layers.split_heads(num_heads)

    attention_bias = Axon.constant(Nx.tensor(0))

    attention_weights =
      Layers.attention_weights(query, key, attention_bias)
      |> Axon.dropout(rate: spec.attention_dropout_rate, name: join(name, "dropout"))

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()

    {attention_output, attention_weights}
  end

  defp self_output(hidden_state, _input_tensor, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp intermediate(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.intermediate_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Axon.activation(spec.activation)
  end

  defp output(hidden_state, attention_output, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
    |> Axon.add(attention_output, name: join(name, "residual"))
  end

  defp pooler(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(index: 0, axis: 1, name: join(name, "head"))
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Axon.tanh(name: join(name, "tanh"))
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
          activation: {"hidden_act", atom()},
          use_qkv_bias: {"qkv_bias", boolean()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
