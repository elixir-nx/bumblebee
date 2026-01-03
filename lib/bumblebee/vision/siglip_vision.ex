defmodule Bumblebee.Vision.SiglipVision do
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
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  The SigLIP model for image encoding.

  ## Architectures

    * `:base` - the base image model

    * `:for_image_classification` - SigLIP vision encoder with a classification
      head. The head consists of a single dense layer on top of the mean-pooled
      patch embeddings

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

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

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    pooled =
      Axon.nx(outputs.hidden_state, fn hidden_state ->
        Nx.mean(hidden_state, axes: [1])
      end)

    logits =
      Axon.dense(pooled, spec.num_labels,
        kernel_initializer: Axon.Initializers.normal(),
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
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp core(inputs, spec) do
    embeddings = embedder(inputs["pixel_values"], spec, name: "embedder")

    encoder_outputs = encoder(embeddings, spec, name: "encoder")

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: "post_norm"
      )

    pooled_state = attention_pooling_head(hidden_state, spec, name: "head")

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    patch_embeddings = patch_embedding(pixel_values, spec, name: join(name, "patch_embedding"))

    num_patches = div(spec.image_size, spec.patch_size) ** 2
    position_ids = position_ids(num_patches)

    position_embeddings =
      Axon.embedding(position_ids, num_patches, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    Axon.add(patch_embeddings, position_embeddings)
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: Axon.Initializers.normal(),
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
      name: join(name, "blocks")
    )
  end

  defp attention_pooling_head(hidden_state, spec, opts) do
    name = opts[:name]

    probe =
      Layers.learned_embeddings(1, spec.hidden_size,
        name: join(name, "probe"),
        initializer: Axon.Initializers.normal()
      )

    attended =
      multihead_attention(probe, hidden_state, spec, name: join(name, "attention"))

    attended = Axon.nx(attended, fn x -> Nx.squeeze(x, axes: [1]) end)

    normed =
      Axon.layer_norm(attended, epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))

    mlp_output =
      normed
      |> Axon.dense(spec.intermediate_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "mlp.intermediate")
      )
      |> Layers.activation(spec.activation)
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "mlp.output")
      )

    Axon.add(attended, mlp_output)
  end

  defp multihead_attention(query, key_value, spec, opts) do
    name = opts[:name]
    num_heads = spec.num_attention_heads
    head_dim = div(spec.hidden_size, num_heads)

    q =
      Axon.dense(query, spec.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "query")
      )

    k =
      Axon.dense(key_value, spec.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "key")
      )

    v =
      Axon.dense(key_value, spec.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "value")
      )

    q = Axon.nx(q, fn x -> reshape_for_attention(x, num_heads, head_dim) end)
    k = Axon.nx(k, fn x -> reshape_for_attention(x, num_heads, head_dim) end)
    v = Axon.nx(v, fn x -> reshape_for_attention(x, num_heads, head_dim) end)

    scale = :math.sqrt(head_dim)

    attention_output =
      Axon.layer(
        fn q, k, v, _opts ->
          # Broadcast q to match k's batch size (for attention pooling head)
          {batch_k, _, _, _} = Nx.shape(k)
          {batch_q, heads, seq_q, head_d} = Nx.shape(q)

          q =
            if batch_q == 1 and batch_k > 1 do
              Nx.broadcast(q, {batch_k, heads, seq_q, head_d})
            else
              q
            end

          scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)
          weights = Axon.Activations.softmax(scores, axis: -1)
          Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
        end,
        [q, k, v],
        name: join(name, "attention"),
        op_name: :attention
      )

    attention_output =
      Axon.nx(attention_output, fn x ->
        {batch, _heads, seq_len, _head_dim} = Nx.shape(x)

        Nx.transpose(x, axes: [0, 2, 1, 3])
        |> Nx.reshape({batch, seq_len, spec.hidden_size})
      end)

    Axon.dense(attention_output, spec.hidden_size,
      kernel_initializer: Axon.Initializers.normal(),
      name: join(name, "output")
    )
  end

  defp reshape_for_attention(x, num_heads, head_dim) do
    {batch, seq_len, _hidden} = Nx.shape(x)
    x |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, %{"model_type" => "siglip", "vision_config" => data}) do
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
        "post_norm" => "vision_model.post_layernorm",
        "head.probe" => %{
          "embeddings" => {
            [{"vision_model.head", "probe"}],
            fn [probe] -> Nx.squeeze(probe, axes: [0]) end
          }
        },
        "head.attention.query" => %{
          "kernel" => {
            [{"vision_model.head.attention", "in_proj_weight"}],
            fn [kernel] ->
              chunk_size = div(Nx.axis_size(kernel, 0), 3)
              kernel = Nx.slice_along_axis(kernel, 0, chunk_size, axis: 0)
              Nx.transpose(kernel)
            end
          },
          "bias" => {
            [{"vision_model.head.attention", "in_proj_bias"}],
            fn [bias] ->
              chunk_size = div(Nx.axis_size(bias, 0), 3)
              Nx.slice_along_axis(bias, 0, chunk_size, axis: 0)
            end
          }
        },
        "head.attention.key" => %{
          "kernel" => {
            [{"vision_model.head.attention", "in_proj_weight"}],
            fn [kernel] ->
              chunk_size = div(Nx.axis_size(kernel, 0), 3)
              kernel = Nx.slice_along_axis(kernel, chunk_size, chunk_size, axis: 0)
              Nx.transpose(kernel)
            end
          },
          "bias" => {
            [{"vision_model.head.attention", "in_proj_bias"}],
            fn [bias] ->
              chunk_size = div(Nx.axis_size(bias, 0), 3)
              Nx.slice_along_axis(bias, chunk_size, chunk_size, axis: 0)
            end
          }
        },
        "head.attention.value" => %{
          "kernel" => {
            [{"vision_model.head.attention", "in_proj_weight"}],
            fn [kernel] ->
              chunk_size = div(Nx.axis_size(kernel, 0), 3)
              kernel = Nx.slice_along_axis(kernel, 2 * chunk_size, chunk_size, axis: 0)
              Nx.transpose(kernel)
            end
          },
          "bias" => {
            [{"vision_model.head.attention", "in_proj_bias"}],
            fn [bias] ->
              chunk_size = div(Nx.axis_size(bias, 0), 3)
              Nx.slice_along_axis(bias, 2 * chunk_size, chunk_size, axis: 0)
            end
          }
        },
        "head.attention.output" => "vision_model.head.attention.out_proj",
        "head.norm" => "vision_model.head.layernorm",
        "head.mlp.intermediate" => "vision_model.head.mlp.fc1",
        "head.mlp.output" => "vision_model.head.mlp.fc2",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end
