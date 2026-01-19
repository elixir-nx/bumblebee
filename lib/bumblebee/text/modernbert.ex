defmodule Bumblebee.Text.ModernBert do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50368,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 8192,
        doc: """
        the maximum sequence length that this model can process. ModernBERT uses RoPE
        (Rotary Position Embedding) instead of absolute position embeddings
        """
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 22,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 1152,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function used in the gated FFN"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for embedding and encoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      classifier_dropout_rate: [
        default: nil,
        doc:
          "the dropout rate for the classification head. If not specified, the value of `:dropout_rate` is used instead"
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
      local_attention_window: [
        default: 128,
        doc: "the window size for local attention layers"
      ],
      layer_types: [
        default: nil,
        doc: """
        a list of layer types for each layer, where each element is either `:sliding_attention`
        (local attention with sliding window) or `:full_attention` (global attention)
        """
      ],
      rotary_embedding_base_local: [
        default: 10_000.0,
        doc: "base for computing rotary embedding frequency for local (sliding) attention layers"
      ],
      rotary_embedding_base: [
        default: 160_000.0,
        doc: "base for computing rotary embedding frequency for global attention layers"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  ModernBERT model family.

  ## Architectures

    * `:base` - plain ModernBERT without any head on top

    * `:for_masked_language_modeling` - ModernBERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - ModernBERT with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - ModernBERT with a token classification
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_masked_language_modeling,
      :for_sequence_classification,
      :for_token_classification
    ]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(%{architecture: :for_sequence_classification}) do
    %{
      "input_ids" => Nx.template({1, 1}, :u32),
      "attention_mask" => Nx.template({1, 1}, :u32)
    }
  end

  def input_template(_spec) do
    %{"input_ids" => Nx.template({1, 1}, :u32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_masked_language_modeling} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> mean_pooling(inputs["attention_mask"])
      |> Axon.dense(spec.hidden_size,
        use_bias: false,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.dense"
      )
      |> Layers.activation(spec.activation)
      |> layer_norm(
        epsilon: spec.layer_norm_epsilon,
        name: "sequence_classification_head.norm"
      )
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "sequence_classification_head.dropout"
      )
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dense(spec.hidden_size,
        use_bias: false,
        kernel_initializer: kernel_initializer(spec),
        name: "token_classification_head.dense"
      )
      |> Layers.activation(spec.activation)
      |> layer_norm(
        epsilon: spec.layer_norm_epsilon,
        name: "token_classification_head.norm"
      )
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "token_classification_head.dropout"
      )
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "token_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(spec) do
    shape = {nil, nil}
    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(inputs["input_ids"], spec, name: "embedder")

    encoder_outputs =
      encoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["position_ids"],
        spec,
        name: "encoder"
      )

    %{
      hidden_state: encoder_outputs.hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(input_ids, spec, opts) do
    name = opts[:name]

    input_ids
    |> Axon.embedding(spec.vocab_size, spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "token_embedding")
    )
    |> layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp encoder(hidden_state, attention_mask, attention_head_mask, position_ids, spec, opts) do
    name = opts[:name]

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(hidden_state)
      end

    layer_types = spec.layer_types || generate_layer_types(spec.num_blocks)

    attention_window_size = fn idx ->
      case Enum.at(layer_types, idx, :sliding_attention) do
        :full_attention ->
          nil

        :sliding_attention ->
          half_window = div(spec.local_attention_window, 2)
          {half_window, half_window}
      end
    end

    rotary_embedding = fn idx ->
      base =
        case Enum.at(layer_types, idx, :sliding_attention) do
          :full_attention -> spec.rotary_embedding_base
          :sliding_attention -> spec.rotary_embedding_base_local
        end

      [
        position_ids: position_ids,
        max_positions: spec.max_positions,
        base: base
      ]
    end

    layer_norm = fn input, name ->
      if String.ends_with?(name, "encoder.blocks.0.self_attention_norm") do
        # The first self-attention norm is skipped.
        input
      else
        layer_norm(input, epsilon: spec.layer_norm_epsilon, name: name)
      end
    end

    outputs =
      Layers.Transformer.blocks(hidden_state,
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        num_blocks: spec.num_blocks,
        num_attention_heads: spec.num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: layer_norm,
        ffn:
          &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
            activation: spec.activation,
            name: &2
          ),
        block_type: :norm_first,
        rotary_embedding: rotary_embedding,
        attention_window_size: attention_window_size,
        query_use_bias: false,
        key_use_bias: false,
        value_use_bias: false,
        output_use_bias: false,
        name: join(name, "blocks")
      )

    hidden_state =
      layer_norm(outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "output_norm")
      )

    %{
      hidden_state: hidden_state,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    }
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        use_bias: false,
        name: join(name, "intermediate")
      )

    gate =
      Axon.dense(hidden_state, intermediate_size, use_bias: false, name: join(name, "gate"))

    hidden_state = Axon.multiply(Layers.activation(intermediate, activation), gate)

    Axon.dense(hidden_state, output_size, use_bias: false, name: join(name, "output"))
  end

  defp mean_pooling(hidden_state, attention_mask) do
    Axon.layer(
      fn hidden_state, attention_mask, _opts ->
        mask = attention_mask |> Nx.as_type(:f32) |> Nx.new_axis(-1)
        sum = Nx.sum(Nx.multiply(hidden_state, mask), axes: [1])
        count = Nx.sum(mask, axes: [1])
        Nx.divide(sum, Nx.max(count, 1.0e-9))
      end,
      [hidden_state, attention_mask]
    )
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.hidden_size,
      use_bias: false,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Layers.activation(spec.activation)
    |> layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Layers.dense_transposed(spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.bias(name: join(name, "bias"))
  end

  defp classifier_dropout_rate(spec) do
    spec.classifier_dropout_rate || spec.dropout_rate
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  # ModernBERT uses LayerNorm without bias
  defp layer_norm(x, opts) do
    name = opts[:name]
    epsilon = opts[:epsilon] || 1.0e-5

    Axon.layer(
      fn x, gamma, _opts ->
        Axon.Layers.layer_norm(x, gamma, Nx.broadcast(0.0, gamma), epsilon: epsilon)
      end,
      [x, Axon.param("weight", fn shape -> {elem(shape, tuple_size(shape) - 1)} end)],
      name: name,
      op_name: :layer_norm
    )
  end

  defp generate_layer_types(num_blocks, global_attn_every_n_layers \\ 3) do
    for i <- 0..(num_blocks - 1) do
      if rem(i, global_attn_every_n_layers) == 0, do: :full_attention, else: :sliding_attention
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      data =
        Map.put_new_lazy(data, "layer_types", fn ->
          pattern = data["global_attn_every_n_layers"] || 3
          num_blocks = data["num_hidden_layers"] || 22

          for i <- 0..(num_blocks - 1) do
            if rem(i, pattern) == 0, do: "full_attention", else: "sliding_attention"
          end
        end)

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_activation", activation()},
          dropout_rate: {"hidden_dropout_prob", optional(number())},
          attention_dropout_rate: {"attention_probs_dropout_prob", optional(number())},
          classifier_dropout_rate: {"classifier_dropout", optional(number())},
          layer_norm_epsilon: {"layer_norm_eps", optional(number())},
          initializer_scale: {"initializer_range", optional(number())},
          local_attention_window: {"local_attention", number()},
          layer_types:
            {"layer_types",
             list(
               mapping(%{
                 "sliding_attention" => :sliding_attention,
                 "full_attention" => :full_attention
               })
             )},
          rotary_embedding_base_local: {"local_rope_theta", optional(number())},
          rotary_embedding_base: {"global_rope_theta", optional(number())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      head_size = div(spec.hidden_size, spec.num_attention_heads)

      qkv_out_template =
        {[spec.num_attention_heads, spec.num_attention_heads, spec.num_attention_heads],
         head_size}

      %{
        "embedder.token_embedding" => "model.embeddings.tok_embeddings",
        "embedder.norm" => "model.embeddings.norm",
        "encoder.blocks.{n}.self_attention.query" =>
          Shared.sliced_dense_params_source(
            "model.layers.{n}.attn.Wqkv",
            qkv_out_template,
            0
          ),
        "encoder.blocks.{n}.self_attention.key" =>
          Shared.sliced_dense_params_source(
            "model.layers.{n}.attn.Wqkv",
            qkv_out_template,
            1
          ),
        "encoder.blocks.{n}.self_attention.value" =>
          Shared.sliced_dense_params_source(
            "model.layers.{n}.attn.Wqkv",
            qkv_out_template,
            2
          ),
        "encoder.blocks.{n}.self_attention.output" => "model.layers.{n}.attn.Wo",
        "encoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.attn_norm",
        "encoder.blocks.{n}.ffn.intermediate" =>
          Shared.sliced_dense_params_source(
            "model.layers.{n}.mlp.Wi",
            {[1, 1], :auto},
            0
          ),
        "encoder.blocks.{n}.ffn.gate" =>
          Shared.sliced_dense_params_source(
            "model.layers.{n}.mlp.Wi",
            {[1, 1], :auto},
            1
          ),
        "encoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.Wo",
        "encoder.blocks.{n}.output_norm" => "model.layers.{n}.mlp_norm",
        "encoder.output_norm" => "model.final_norm",
        "language_modeling_head.dense" => "head.dense",
        "language_modeling_head.norm" => "head.norm",
        "language_modeling_head.output" => "model.embeddings.tok_embeddings",
        "language_modeling_head.bias" => "decoder",
        "sequence_classification_head.dense" => "head.dense",
        "sequence_classification_head.norm" => "head.norm",
        "sequence_classification_head.output" => "classifier",
        "token_classification_head.dense" => "head.dense",
        "token_classification_head.norm" => "head.norm",
        "token_classification_head.output" => "classifier"
      }
    end
  end
end
