defmodule Bumblebee.Multimodal.LayoutLm do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 30522,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 1024,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      max_spatial_positions: [
        default: 1024,
        doc: """
        the maximum value of the spatial position embedding. Typically this is set to a large value
        just in case, such as 512, 1024, or 2048
        """
      ],
      max_token_types: [
        default: 2,
        doc: """
        the vocabulary size of the token type embedding (also referred to as segment embedding).
        This corresponds to how many different token groups can be distinguished in the input
        """
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
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      intermediate_size: [
        default: 3072,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the decoder.
        If not specified, defaults to 4 times `:hidden_size`
        """
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embedding and encoder"
      ],
      attention_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for attention weights"
      ],
      classifier_dropout_rate: [
        default: nil,
        doc:
          "the dropout rate for the classification head. If not specified, the value of `:dropout_rate` is used instead"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      layer_norm_epsilon: [
        default: 1.0e-12,
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
  LayoutLM Model family.

  ## Architectures

     * `:base` - plain LayoutLM without any head on top

    * `:for_masked_language_modeling` - LayoutLM with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - LayoutLM with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - LayoutLM with a token classification
      head. The head returns logits for each token in the original
      sequence

    * `:for_question_answering` - LayoutLM with a span classification head.
      The head returns logits for the span start and end positions

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"token_type_ids"` - `{batch_size, sequence_length}`

      Mask distinguishing groups in the input sequence. This is used
      in when the input sequence is a semantically a pair of sequences.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"bounding_box"` - `{batch_size, sequence_length, 4}`

    Bounding boxes of each input sequence token. Each bounding box is
    `{x0, y0, x1, y1}` where `{x0, y0}` is the upper left corner and
    `{x1, y1}` is the lower right corner.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [LayoutLM: LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
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
      :for_token_classification,
      :for_question_answering
    ]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(%{architecture: :for_multiple_choice}) do
    %{"input_ids" => Nx.template({1, 1, 1}, :u32)}
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
      outputs.pooled_state
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

  def model(%__MODULE__{architecture: :for_question_answering} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, 2,
        kernel_initializer: kernel_initializer(spec),
        name: "question_answering_head.output"
      )

    {start_logits, end_logits} = Layers.split_pair(logits)

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(spec, opts \\ []) do
    shape = Keyword.get(opts, :shape, {nil, nil})

    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}
    bounding_box_shape = Tuple.append(shape, 4)

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: shape),
      Axon.input("bounding_box", optional: true, shape: bounding_box_shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("token_type_ids", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(
        inputs["input_ids"],
        inputs["bounding_box"],
        inputs["position_ids"],
        inputs["token_type_ids"],
        spec,
        name: "embedder"
      )

    encoder_outputs =
      encoder(embeddings, inputs["attention_mask"], inputs["attention_head_mask"], spec,
        name: "encoder"
      )

    pooled_state = pooler(encoder_outputs.hidden_state, spec, name: "pooler")

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(input_ids, bounding_box, position_ids, token_type_ids, spec, opts) do
    name = opts[:name]

    bounding_box =
      Layers.default bounding_box do
        Layers.default_bounding_box(input_ids)
      end

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_ids)
      end

    token_type_ids =
      Layers.default token_type_ids do
        Layers.default_token_type_ids(input_ids)
      end

    inputs_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        name: join(name, "token_embedding")
      )

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    token_type_embeddings =
      Axon.embedding(token_type_ids, spec.max_token_types, spec.hidden_size,
        name: join(name, "token_type_embedding")
      )

    # TODO: Explicitly tie these weights

    left_position_embeddings =
      bounding_box
      |> Axon.nx(& &1[[.., .., 0]])
      |> Axon.embedding(spec.max_spatial_positions, spec.hidden_size,
        name: join(name, "x_position_embedding")
      )

    right_position_embeddings =
      bounding_box
      |> Axon.nx(& &1[[.., .., 2]])
      |> Axon.embedding(spec.max_spatial_positions, spec.hidden_size,
        name: join(name, "x_position_embedding")
      )

    upper_position_embeddings =
      bounding_box
      |> Axon.nx(& &1[[.., .., 1]])
      |> Axon.embedding(spec.max_spatial_positions, spec.hidden_size,
        name: join(name, "y_position_embedding")
      )

    lower_position_embeddings =
      bounding_box
      |> Axon.nx(& &1[[.., .., 3]])
      |> Axon.embedding(spec.max_spatial_positions, spec.hidden_size,
        name: join(name, "y_position_embedding")
      )

    h_position_embeddings =
      bounding_box
      |> Axon.nx(fn x -> Nx.subtract(x[[.., .., 3]], x[[.., .., 1]]) end)
      |> Axon.embedding(spec.max_spatial_positions, spec.hidden_size,
        name: join(name, "h_position_embedding")
      )

    w_position_embeddings =
      bounding_box
      |> Axon.nx(fn x -> Nx.subtract(x[[.., .., 2]], x[[.., .., 0]]) end)
      |> Axon.embedding(spec.max_spatial_positions, spec.hidden_size,
        name: join(name, "w_position_embedding")
      )

    embeddings =
      Axon.add([
        inputs_embeddings,
        position_embeddings,
        token_type_embeddings,
        left_position_embeddings,
        right_position_embeddings,
        upper_position_embeddings,
        lower_position_embeddings,
        h_position_embeddings,
        w_position_embeddings
      ])

    embeddings
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate)
  end

  defp encoder(hidden_state, attention_mask, attention_head_mask, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: spec.dropout_rate,
      attention_dropout_rate: spec.attention_dropout_rate,
      layer_norm: [
        epsilon: spec.layer_norm_epsilon
      ],
      ffn: [
        intermediate_size: spec.intermediate_size,
        activation: spec.activation
      ],
      output_hidden_states: spec.output_hidden_states,
      output_attentions: spec.output_attentions,
      name: join(name, "blocks")
    )
  end

  defp pooler(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(index: 0, axis: 1, name: join(name, "head"))
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.tanh()
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    # TODO: use a shared parameter with embeddings.word_embeddings.kernel
    # if spec.tie_word_embeddings is true (relevant for training)

    hidden_state
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "dense")
    )
    |> Layers.activation(spec.activation, name: join(name, "activation"))
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    # We reuse the kernel of input embeddings and add bias for each token
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

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          max_token_types: {"type_vocab_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          classifier_dropout_rate: {"classifier_dropout", optional(number())},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "layoutlm.embeddings.word_embeddings",
        "embedder.position_embedding" => "layoutlm.embeddings.position_embeddings",
        "embedder.token_type_embedding" => "layoutlm.embeddings.token_type_embeddings",
        "embedder.x_position_embedding" => "layoutlm.embeddings.x_position_embeddings",
        "embedder.y_position_embedding" => "layoutlm.embeddings.y_position_embeddings",
        "embedder.h_position_embedding" => "layoutlm.embeddings.h_position_embeddings",
        "embedder.w_position_embedding" => "layoutlm.embeddings.w_position_embeddings",
        "embedder.norm" => "layoutlm.embeddings.LayerNorm",
        "encoder.blocks.{n}.self_attention.query" =>
          "layoutlm.encoder.layer.{n}.attention.self.query",
        "encoder.blocks.{n}.self_attention.key" =>
          "layoutlm.encoder.layer.{n}.attention.self.key",
        "encoder.blocks.{n}.self_attention.value" =>
          "layoutlm.encoder.layer.{n}.attention.self.value",
        "encoder.blocks.{n}.self_attention.output" =>
          "layoutlm.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.self_attention_norm" =>
          "layoutlm.encoder.layer.{n}.attention.output.LayerNorm",
        "encoder.blocks.{n}.cross_attention.query" =>
          "layoutlm.encoder.layer.{n}.crossattention.self.query",
        "encoder.blocks.{n}.cross_attention.key" =>
          "layoutlm.encoder.layer.{n}.crossattention.self.key",
        "encoder.blocks.{n}.cross_attention.value" =>
          "layoutlm.encoder.layer.{n}.crossattention.self.value",
        "encoder.blocks.{n}.cross_attention.output" =>
          "layoutlm.encoder.layer.{n}.crossattention.output.dense",
        "encoder.blocks.{n}.cross_attention_norm" =>
          "layoutlm.encoder.layer.{n}.crossattention.output.LayerNorm",
        "encoder.blocks.{n}.ffn.intermediate" => "layoutlm.encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => "layoutlm.encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => "layoutlm.encoder.layer.{n}.output.LayerNorm",
        "pooler.output" => "layoutlm.pooler.dense",
        "language_modeling_head.dense" => "cls.predictions.transform.dense",
        "language_modeling_head.norm" => "cls.predictions.transform.LayerNorm",
        "language_modeling_head.output" => "cls.predictions.decoder",
        "language_modeling_head.bias" => "cls.predictions",
        "sequence_classification_head.output" => "classifier",
        "token_classification_head.output" => "classifier",
        "multiple_choice_head.output" => "classifier",
        "question_answering_head.output" => "qa_outputs"
      }
    end
  end
end
