defmodule Bumblebee.Text.Bert do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :label2id, :num_labels]

  @moduledoc """
  Models based on the BERT architecture.

  ## Architectures

    * `:base` - plain BERT without any head on top

    * `:for_masked_language_modeling` - BERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_causal_language_modeling` - BERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - BERT with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - BERT with a token classification
      head. The head returns logits for each token in the original
      sequence

    * `:for_question_answering` - BERT with a span classification head.
      The head returns logits for the span start and end positions

    * `:for_multiple_choice` - BERT with a multiple choice prediction
      head. Each input in the batch consists of several sequences to
      choose from and the model returns logits corresponding to those
      choices

    * `:for_next_sentence_prediction` - BERT with a next sentence
      prediction head. The head returns logits predicting whether the
      second sentence is random or in context

    * `:for_pre_training` - BERT with both MLM and NSP heads as done
      during the pre-training

  ## Inputs

    * `"input_ids"` - indices of input sequence tokens in the vocabulary

    * `"attention_mask"` - a mask indicating which tokens to attend to.
      This is used to ignore padding tokens, which are added when
      processing a batch of sequences with different length

    * `"token_type_ids"` - a mask distinguishing groups in the input
      sequence. This is used in when the input sequence is a semantically
      a pair of sequences

    * `"position_ids"` - indices of positions of each input sequence
      tokens in the position embeddings

    * `"head_mask"` - a mask to nullify selected heads of the self-attention
      blocks

  ## Configuration

    * `:vocab_size` - vocabulary size of the model. Defines the number
      of distinct tokens that can be represented by the in model input
      and output. Defaults to `30522`

    * `:hidden_size` - dimensionality of the encoder layers and the
      pooler layer. Defaults to `768`

    * `:num_hidden_layers` - the number of hidden layers in the
      Transformer encoder. Defaults to `12`

    * `:num_attention_heads` - the number of attention heads for each
      attention layer in the Transformer encoder. Defaults to `12`

    * `:intermediate_size` - dimensionality of the "intermediate"
      (often named feed-forward) layer in the Transformer encoder.
      Defaults to `3072`

    * `:hidden_act` - the activation function in the encoder and
      pooler. Defaults to `:gelu`

    * `:hidden_dropout_prob` - the dropout probability for all fully
      connected layers in the embeddings, encoder, and pooler. Defaults
      to `0.1`

    * `:attention_probs_dropout_prob` - the dropout probability for
      attention probabilities. Defaults to `0.1`

    * `:max_position_embeddings` - the maximum sequence length that this
      model might ever be used with. Typically set this to something
      large just in case (e.g. 512 or 1024 or 2048). Defaults to `512`

    * `:type_vocab_size` - the vocabulary size of the `token_type_ids`
      passed as part of model input. Defaults to `2`

    * `:initializer_range` - the standard deviation of the normal
      initializer used for initializing kernel parameters. Defaults
      to `0.02`

    * `:layer_norm_eps` - the epsilon used by the layer normalization
      layers. Defaults to `1.0e-12`

    * `:classifier_dropout` - the dropout ratio for the classification
      head. If not specified, the value of `:hidden_dropout_prob` is
      used instead

  ### Common options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              vocab_size: 30522,
              hidden_size: 768,
              num_hidden_layers: 12,
              num_attention_heads: 12,
              intermediate_size: 3072,
              hidden_act: :gelu,
              hidden_dropout_prob: 0.1,
              attention_probs_dropout_prob: 0.1,
              max_position_embeddings: 512,
              type_vocab_size: 2,
              initializer_range: 0.02,
              layer_norm_eps: 1.0e-12,
              classifier_dropout: nil
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

  @impl true
  def architectures(),
    do: [
      :base,
      :for_masked_language_modeling,
      :for_causal_language_modeling,
      :for_sequence_classification,
      :for_token_classification,
      :for_question_answering,
      :for_multiple_choice,
      :for_next_sentence_prediction,
      :for_pre_training
    ]

  @impl true
  def base_model_prefix(), do: "bert"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs({nil, 11}, config)
    |> bert(config)
    |> Bumblebee.Utils.Model.output(config)
  end

  def model(%__MODULE__{architecture: :for_masked_language_modeling} = config) do
    outputs = inputs({nil, 9}, config) |> bert(config, name: "bert")

    logits = lm_prediction_head(outputs.last_hidden_state, config, name: "cls.predictions")

    Bumblebee.Utils.Model.output(
      %{
        logits: logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = config) do
    outputs = inputs({nil, 8}, config) |> bert(config, name: "bert")

    logits = lm_prediction_head(outputs.last_hidden_state, config, name: "cls.predictions")

    Bumblebee.Utils.Model.output(
      %{
        logits: logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = config) do
    outputs = inputs({nil, 11}, config) |> bert(config, name: "bert")

    logits =
      outputs.pooler_output
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(config.num_labels,
        kernel_initializer: kernel_initializer(config),
        name: "classifier"
      )

    Bumblebee.Utils.Model.output(
      %{
        logits: logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_token_classification} = config) do
    outputs = inputs({nil, 13}, config) |> bert(config, name: "bert")

    logits =
      outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(config.num_labels,
        kernel_initializer: kernel_initializer(config),
        name: "classifier"
      )

    Bumblebee.Utils.Model.output(
      %{
        logits: logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_question_answering} = config) do
    outputs = inputs({nil, 16}, config) |> bert(config, name: "bert")

    logits =
      outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(2,
        kernel_initializer: kernel_initializer(config),
        name: "qa_outputs"
      )

    {start_logits, end_logits} = Axon.split(logits, 2, axis: -1)
    start_logits = flatten_trailing(start_logits)
    end_logits = flatten_trailing(end_logits)

    Bumblebee.Utils.Model.output(
      %{
        start_logits: start_logits,
        end_logits: end_logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_multiple_choice} = config) do
    inputs = inputs({nil, nil, 35}, config)

    flat_inputs =
      Map.new(inputs, fn {key, input} -> {key, Layers.flatten_leading_layer(input)} end)

    outputs = bert(flat_inputs, config, name: "bert")

    logits =
      outputs.pooler_output
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(1,
        kernel_initializer: kernel_initializer(config),
        name: "classifier"
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

    Bumblebee.Utils.Model.output(
      %{
        logits: logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_next_sentence_prediction} = config) do
    outputs = inputs({nil, 38}, config) |> bert(config, name: "bert")

    logits =
      outputs.pooler_output
      |> Axon.dense(2,
        kernel_initializer: kernel_initializer(config),
        name: "cls.seq_relationship"
      )

    Bumblebee.Utils.Model.output(
      %{
        logits: logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  def model(%__MODULE__{architecture: :for_pre_training} = config) do
    outputs = inputs({nil, 8}, config) |> bert(config, name: "bert")

    prediction_logits =
      lm_prediction_head(outputs.last_hidden_state, config, name: "cls.predictions")

    seq_relationship_logits =
      Axon.dense(outputs.pooler_output, 2,
        kernel_initializer: kernel_initializer(config),
        name: "cls.seq_relationship"
      )

    Bumblebee.Utils.Model.output(
      %{
        prediction_logits: prediction_logits,
        seq_relationship_logits: seq_relationship_logits,
        hidden_states: outputs.hidden_states,
        attentions: outputs.attentions
      },
      config
    )
  end

  defp inputs(input_shape, config) do
    %{
      "input_ids" => Axon.input(input_shape, "input_ids"),
      "attention_mask" =>
        Axon.input(input_shape, "attention_mask",
          default: fn inputs -> Nx.broadcast(1, inputs["input_ids"]) end
        ),
      "token_type_ids" =>
        Axon.input(input_shape, "token_type_ids",
          default: fn inputs -> Nx.broadcast(0, inputs["input_ids"]) end
        ),
      "position_ids" =>
        Axon.input(input_shape, "position_ids",
          default: fn inputs -> Nx.iota(inputs["input_ids"], axis: -1) end
        ),
      "head_mask" =>
        Axon.input({config.num_hidden_layers, config.num_attention_heads}, "head_mask",
          default: fn _inputs ->
            Nx.broadcast(1, {config.num_hidden_layers, config.num_attention_heads})
          end
        )
    }
  end

  defp bert(inputs, config, opts \\ []) do
    name = opts[:name]

    hidden_state =
      embeddings(inputs["input_ids"], inputs["token_type_ids"], inputs["position_ids"], config,
        name: join(name, "embeddings")
      )

    {last_hidden_state, hidden_states, attentions} =
      encoder(hidden_state, inputs["attention_mask"], inputs["head_mask"], config,
        name: join(name, "encoder")
      )

    pooler_output = pooler(last_hidden_state, config, name: join(name, "pooler"))

    %{
      last_hidden_state: last_hidden_state,
      pooler_output: pooler_output,
      hidden_states: hidden_states,
      attentions: attentions
    }
  end

  defp embeddings(input_ids, token_type_ids, position_ids, config, opts) do
    name = opts[:name]

    inputs_embeds =
      Axon.embedding(input_ids, config.vocab_size, config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "word_embeddings")
      )

    position_embeds =
      Axon.embedding(position_ids, config.max_position_embeddings, config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "position_embeddings")
      )

    token_type_embeds =
      Axon.embedding(token_type_ids, config.type_vocab_size, config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "token_type_embeddings")
      )

    Axon.add([inputs_embeds, position_embeds, token_type_embeds])
    |> Axon.layer_norm(
      epsilon: config.layer_norm_eps,
      name: join(name, "LayerNorm"),
      channel_index: 2
    )
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
  end

  defp encoder(hidden_state, attention_mask, head_mask, config, opts) do
    name = opts[:name]

    encoder_layers(hidden_state, attention_mask, head_mask, config, name: join(name, "layer"))
  end

  defp encoder_layers(hidden_state, attention_mask, head_mask, config, opts) do
    name = opts[:name]

    for idx <- 0..(config.num_hidden_layers - 1), reduce: {hidden_state, {hidden_state}, {}} do
      {hidden_state, hidden_states, attentions} ->
        layer_head_mask = Axon.nx(head_mask, & &1[idx])

        {hidden_state, attention} =
          bert_layer(hidden_state, attention_mask, layer_head_mask, config, name: join(name, idx))

        {
          hidden_state,
          Tuple.append(hidden_states, hidden_state),
          Tuple.append(attentions, attention)
        }
    end
  end

  defp bert_layer(hidden_state, attention_mask, layer_head_mask, config, opts) do
    name = opts[:name]

    {attention_output, attention} =
      attention(hidden_state, attention_mask, layer_head_mask, config,
        name: join(name, "attention")
      )

    hidden_state = intermediate(attention_output, config, name: join(name, "intermediate"))
    hidden_state = output(hidden_state, attention_output, config, name: join(name, "output"))

    {hidden_state, attention}
  end

  defp attention(hidden_state, attention_mask, layer_head_mask, config, opts) do
    name = opts[:name]

    {attention_output, attention} =
      self_attention(hidden_state, attention_mask, layer_head_mask, config,
        name: join(name, "self")
      )

    hidden_state = self_output(attention_output, hidden_state, config, name: join(name, "output"))

    {hidden_state, attention}
  end

  defp self_attention(hidden_state, attention_mask, layer_head_mask, config, opts) do
    name = opts[:name]

    head_dim = div(config.hidden_size, config.num_attention_heads)

    query_states =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "query")
      )
      |> Axon.reshape({:auto, config.num_attention_heads, head_dim})

    value_states =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "value")
      )
      |> Axon.reshape({:auto, config.num_attention_heads, head_dim})

    key_states =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "key")
      )
      |> Axon.reshape({:auto, config.num_attention_heads, head_dim})

    attention_bias = Axon.layer(&Layers.attention_bias/2, [attention_mask])

    attention_weights =
      Axon.layer(&Layers.attention_weights/4, [query_states, key_states, attention_bias])

    attention_weights =
      Axon.dropout(attention_weights,
        rate: config.attention_probs_dropout_prob,
        name: join(name, "dropout")
      )

    attention_weights =
      Axon.layer(&Layers.apply_layer_head_mask/3, [attention_weights, layer_head_mask])

    attention_output = Axon.layer(&Layers.attention_output/3, [attention_weights, value_states])

    attention_output =
      Axon.reshape(attention_output, {:auto, config.num_attention_heads * head_dim})

    {attention_output, attention_weights}
  end

  defp self_output(hidden_state, input, config, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
    |> Axon.add(input)
    |> Axon.layer_norm(
      epsilon: config.layer_norm_eps,
      name: join(name, "LayerNorm"),
      channel_index: 2
    )
  end

  defp intermediate(hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(config.intermediate_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Layers.activation_layer(config.hidden_act, name: join(name, "activation"))
  end

  defp output(hidden_state, attention_output, config, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dropout"))
    |> Axon.add(attention_output)
    |> Axon.layer_norm(
      epsilon: config.layer_norm_eps,
      name: join(name, "LayerNorm"),
      channel_index: 2
    )
  end

  defp pooler(hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token_layer(index: 0, axis: 1, name: join(name, "head"))
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Axon.tanh()
  end

  defp lm_prediction_head(hidden_state, config, opts) do
    name = opts[:name]

    # TODO: use a shared parameter with embeddings.word_embeddings.kernel
    # if config.tie_word_embeddings is true (relevant for training)

    hidden_state
    |> lm_prediction_head_transform(config, name: join(name, "transform"))
    # We reuse the kernel of input embeddings and add bias for each token
    |> Layers.dense_transposed_layer(config.vocab_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "decoder")
    )
    |> Axon.bias(name: name)
  end

  defp lm_prediction_head_transform(hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Layers.activation_layer(config.hidden_act, name: join(name, "activation"))
    |> Axon.layer_norm(
      epsilon: config.layer_norm_eps,
      name: join(name, "LayerNorm"),
      channel_index: 2
    )
  end

  defp flatten_trailing(%Axon{} = x) do
    Axon.nx(x, fn x ->
      shape = Nx.shape(x)
      rank = tuple_size(shape)

      shape =
        shape
        |> Tuple.delete_at(rank - 1)
        |> put_elem(rank - 2, :auto)

      Nx.reshape(x, shape)
    end)
  end

  defp classifier_dropout_rate(config) do
    config.classifier_dropout || config.hidden_dropout_prob
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.initializer_range)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["hidden_act"])
      |> Shared.convert_common()
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
