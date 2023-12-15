defmodule Bumblebee.Text.Albert do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 30000,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 512,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      max_token_types: [
        default: 2,
        doc: """
        the vocabulary size of the token type embedding (also referred to as segment embedding).
        This corresponds to how many different token groups can be distinguished in the input
        """
      ],
      embedding_size: [
        default: 128,
        doc: "the dimensionality of all input embeddings"
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 12,
        doc: """
        the number of blocks in the encoder. Note that each block contains `:block_depth`
        Transformer blocks
        """
      ],
      num_groups: [
        default: 1,
        doc: "the number of groups of encoder blocks. Parameters in the same group are shared"
      ],
      block_depth: [
        default: 1,
        doc: "the number of Transformer blocks in each encoder block"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 16384,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
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
  ALBERT model family.

  ## Architectures

    * `:base` - plain ALBERT without any head on top

    * `:for_masked_language_modeling` - ALBERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - ALBERT with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - ALBERT with a token classification
      head. The head returns logits for each token in the original
      sequence

    * `:for_question_answering` - ALBERT with a span classification head.
      The head returns logits for the span start and end positions

    * `:for_multiple_choice` - ALBERT with a multiple choice prediction
      head. Each input in the batch consists of several sequences to
      choose from and the model returns logits corresponding to those
      choices

    * `:for_pre_training` - ALBERT with both MLM and NSP heads as done
      during the pre-training

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

  ### Exceptions

  The `:for_multiple_choice` model accepts groups of sequences, so the
  expected sequence shape is `{batch_size, num_choices, sequence_length}`.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

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
      :for_question_answering,
      :for_multiple_choice,
      :for_pre_training
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
    inputs()
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_masked_language_modeling} = spec) do
    outputs = core(inputs(), spec)

    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    outputs = core(inputs(), spec)

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
    outputs = core(inputs(), spec)

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
    outputs = core(inputs(), spec)

    logits =
      Axon.dense(outputs.hidden_state, 2,
        kernel_initializer: kernel_initializer(spec),
        name: "question_answering_head.output"
      )

    start_logits = Axon.nx(logits, & &1[[.., .., 0]])
    end_logits = Axon.nx(logits, & &1[[.., .., 1]])

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_multiple_choice} = spec) do
    inputs = inputs({nil, nil, nil})

    group_inputs = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]

    flat_inputs =
      Enum.reduce(group_inputs, inputs, fn name, inputs ->
        Map.update!(inputs, name, &Layers.flatten_leading/1)
      end)

    outputs = core(flat_inputs, spec)

    logits =
      outputs.pooled_state
      |> Axon.dropout(rate: classifier_dropout_rate(spec), name: "multiple_choice_head.dropout")
      |> Axon.dense(1,
        kernel_initializer: kernel_initializer(spec),
        name: "multiple_choice_head.output"
      )

    # The final shape depends on the dynamic batch size and number
    # of choices, so we do a custom reshape at runtime
    logits =
      Axon.layer(
        fn logits, input_ids, _opts ->
          num_choices = Nx.axis_size(input_ids, 1)
          Nx.reshape(logits, {:auto, num_choices})
        end,
        [logits, inputs["input_ids"]]
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(shape \\ {nil, nil}) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: shape),
      Axon.input("attention_mask", shape: shape, optional: true),
      Axon.input("token_type_ids", shape: shape, optional: true),
      Axon.input("position_ids", shape: shape, optional: true)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["input_ids"], inputs["position_ids"], inputs["token_type_ids"], spec,
        name: "embedder"
      )

    {hidden_state, hidden_states, attentions} =
      encoder(embeddings, inputs["attention_mask"], spec, name: join(name, "encoder"))

    pooled_state = pooler(hidden_state, spec, name: join(name, "pooler"))

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: hidden_states,
      attentions: attentions
    }
  end

  defp embedder(input_ids, position_ids, token_type_ids, spec, opts) do
    name = opts[:name]

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_ids)
      end

    token_type_ids =
      Layers.default token_type_ids do
        Layers.default_token_type_ids(input_ids)
      end

    inputs_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.embedding_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.embedding_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "position_embedding")
      )

    token_type_embeddings =
      Axon.embedding(token_type_ids, spec.max_token_types, spec.embedding_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_type_embedding")
      )

    Axon.add([inputs_embeddings, position_embeddings, token_type_embeddings])
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp encoder(hidden_state, attention_mask, spec, opts) do
    name = opts[:name]

    hidden_state =
      Axon.dense(hidden_state, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "embedding_projection")
      )

    hidden_states = Layers.maybe_container({hidden_state}, spec.output_hidden_states)
    attentions = Layers.maybe_container({}, spec.output_attentions)

    for block_idx <- 0..(spec.num_blocks - 1),
        inner_idx <- 0..(spec.block_depth - 1),
        reduce: {hidden_state, hidden_states, attentions} do
      {hidden_state, hidden_states, attentions} ->
        group_idx = div(block_idx, div(spec.num_blocks, spec.num_groups))

        name = name |> join("groups") |> join(group_idx) |> join("blocks") |> join(inner_idx)

        # TODO: wrap encoder block in a layer_drop combinator
        {hidden_state, attention, _cross_attention, _block_cache, _position_bias} =
          Layers.Transformer.block(hidden_state,
            attention_mask: attention_mask,
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
            name: name
          )

        {
          hidden_state,
          Layers.append(hidden_states, hidden_state),
          Layers.append(attentions, attention)
        }
    end
  end

  defp pooler(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(axis: 1)
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
    |> Axon.tanh(name: join(name, "tanh"))
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    # TODO: use a shared parameter with embeddings.word_embeddings.kernel
    # if spec.tie_word_embeddings is true (relevant for training)

    hidden_state
    |> Axon.dense(spec.embedding_size,
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
          embedding_size: {"embedding_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_groups: {"num_hidden_groups", number()},
          block_depth: {"inner_group_num", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          classifier_dropout_rate: {"classifier_dropout_prob", optional(number())},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "albert.embeddings.word_embeddings",
        "embedder.position_embedding" => "albert.embeddings.position_embeddings",
        "embedder.token_type_embedding" => "albert.embeddings.token_type_embeddings",
        "embedder.norm" => "albert.embeddings.LayerNorm",
        "encoder.embedding_projection" => "albert.encoder.embedding_hidden_mapping_in",
        "encoder.groups.{n}.blocks.{m}.self_attention.query" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.attention.query",
        "encoder.groups.{n}.blocks.{m}.self_attention.key" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.attention.key",
        "encoder.groups.{n}.blocks.{m}.self_attention.value" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.attention.value",
        "encoder.groups.{n}.blocks.{m}.self_attention.output" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.attention.dense",
        "encoder.groups.{n}.blocks.{m}.self_attention_norm" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.attention.LayerNorm",
        "encoder.groups.{n}.blocks.{m}.ffn.intermediate" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.ffn",
        "encoder.groups.{n}.blocks.{m}.ffn.output" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.ffn_output",
        "encoder.groups.{n}.blocks.{m}.output_norm" =>
          "albert.encoder.albert_layer_groups.{n}.albert_layers.{m}.full_layer_layer_norm",
        "pooler.output" => "albert.pooler",
        "language_modeling_head.dense" => "predictions.dense",
        "language_modeling_head.norm" => "predictions.LayerNorm",
        "language_modeling_head.output" => "predictions.decoder",
        "sequence_classification_head.output" => "classifier",
        "token_classification_head.output" => "classifier",
        "multiple_choice_head.output" => "classifier",
        "question_answering_head.output" => "qa_outputs"
      }
    end
  end
end
