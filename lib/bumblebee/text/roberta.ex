defmodule Bumblebee.Text.Roberta do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :label2id, :num_labels]

  @moduledoc """
  Models based on the RoBERTa architecture.

  ## Architectures

    * `:base` - plain RoBERTa without any head on top

    * `:for_masked_language_modeling` - RoBERTa with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - RoBERTa with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - RoBERTa with a token classification
      head. The head returns logits for each token in the original
      sequence

    * `:for_question_answering` - RoBERTa with a span classification head.
      The head returns logits for the span start and end positions

    * `:for_multiple_choice` - RoBERTa with a multiple choice prediction
      head. Each input in the batch consists of several sequences to
      choose from and the model returns logits corresponding to those
      choices

    * `:for_causal_language_modeling` - RoBERTa with a language modeling
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, seq_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, seq_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"token_type_ids"` - `{batch_size, seq_length}`

      Mask distinguishing groups in the input sequence. This is used
      in when the input sequence is a semantically a pair of sequences.

    * `"position_ids"` - `{batch_size, seq_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"head_mask"` - `{encoder_layers, encoder_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

  ### Exceptions

  The `:for_multiple_choice` model accepts groups of sequences, so the
  expected sequence shape is `{batch_size, num_choices, seq_length}`.

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

    * `:pad_token_id` - the padding identifier for <pad> token, equals to 1

  ### Common options

  #{Bumblebee.Shared.common_config_docs(@common_keys)}
  """

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              vocab_size: 50265,
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
              classifier_dropout: nil,
              pad_token_id: 1
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
      :for_multiple_choice
    ]

  @impl true
  def base_model_prefix(), do: "roberta"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def input_template(%{architecture: :for_multiple_choice}) do
    %{"input_ids" => Nx.template({1, 1, 1}, :s64)}
  end

  def input_template(_config) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs(config)
    |> roberta(config)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_masked_language_modeling} = config) do
    outputs = inputs(config) |> roberta(config, name: "roberta")

    logits = lm_prediction_head(outputs.last_hidden_state, config, name: "lm_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = config) do
    outputs = inputs(config) |> roberta(config, name: "roberta")

    logits = lm_prediction_head(outputs.last_hidden_state, config, name: "lm_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = config) do
    outputs = inputs(config) |> roberta(config, name: "roberta")

    logits =
      outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: "classifier.dense"
      )
      |> Axon.tanh()
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(config.num_labels,
        kernel_initializer: kernel_initializer(config),
        name: "classifier.out_proj"
      )

    Layers.output(%{
      logits: Axon.nx(logits, fn x -> x[[0..-1//1, 0, 0..-1//1]] end),
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = config) do
    outputs = inputs(config) |> roberta(config, name: "roberta")

    logits =
      outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(config.num_labels,
        kernel_initializer: kernel_initializer(config),
        name: "classifier"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_question_answering} = config) do
    outputs = inputs(config) |> roberta(config, name: "roberta")

    logits =
      outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(config), name: "dropout")
      |> Axon.dense(2,
        kernel_initializer: kernel_initializer(config),
        name: "qa_outputs"
      )

    {start_logits, end_logits} = Layers.split_pair(logits)

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_multiple_choice} = config) do
    inputs = inputs(config, {nil, nil, nil})

    group_inputs = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]

    flat_inputs =
      Enum.reduce(group_inputs, inputs, fn name, inputs ->
        Map.update!(inputs, name, &Layers.flatten_leading/1)
      end)

    outputs = roberta(flat_inputs, config, name: "roberta")

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

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(config, shape \\ {nil, nil}) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: shape),
      Axon.input("attention_mask", shape: shape, optional: true),
      Axon.input("token_type_ids", shape: shape, optional: true),
      Axon.input("position_ids", shape: shape, optional: true),
      Axon.input("head_mask",
        shape: {config.num_hidden_layers, config.num_attention_heads},
        optional: true
      )
    ])
  end

  defp roberta(inputs, config, opts \\ []) do
    name = opts[:name]

    input_ids = inputs["input_ids"]

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(input_ids)
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        # Position numbers begin at padding token id + 1
        Layers.default_position_ids(input_ids, offset: config.pad_token_id + 1)
      end

    token_type_ids =
      Layers.default inputs["token_type_ids"] do
        Layers.default_token_type_ids(input_ids)
      end

    hidden_state =
      embeddings(input_ids, position_ids, token_type_ids, config, name: join(name, "embeddings"))

    {last_hidden_state, hidden_states, attentions} =
      encoder(hidden_state, attention_mask, inputs["head_mask"], config,
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

  defp embeddings(input_ids, position_ids, token_type_ids, config, opts) do
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

    hidden_states = Layers.maybe_container({hidden_state}, config.output_hidden_states)
    attentions = Layers.maybe_container({}, config.output_attentions)

    for idx <- 0..(config.num_hidden_layers - 1),
        reduce: {hidden_state, hidden_states, attentions} do
      {hidden_state, hidden_states, attentions} ->
        layer_head_mask = Axon.nx(head_mask, & &1[idx])

        {hidden_state, attention} =
          roberta_layer(hidden_state, attention_mask, layer_head_mask, config,
            name: join(name, idx)
          )

        {
          hidden_state,
          Layers.append(hidden_states, hidden_state),
          Layers.append(attentions, attention)
        }
    end
  end

  defp roberta_layer(hidden_state, attention_mask, layer_head_mask, config, opts) do
    name = opts[:name]

    {attention_outputs, attention} =
      attention(hidden_state, attention_mask, layer_head_mask, config,
        name: join(name, "attention")
      )

    hidden_state = intermediate(attention_outputs, config, name: join(name, "intermediate"))
    hidden_state = output(hidden_state, attention_outputs, config, name: join(name, "output"))

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

    num_heads = config.num_attention_heads

    query =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "query")
      )
      |> Layers.split_heads(num_heads)

    value =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "value")
      )
      |> Layers.split_heads(num_heads)

    key =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "key")
      )
      |> Layers.split_heads(num_heads)

    attention_mask = Layers.expand_attention_mask(attention_mask)
    attention_bias = Layers.attention_bias(attention_mask)

    attention_weights =
      Layers.attention_weights(query, key, attention_bias)
      |> Axon.dropout(rate: config.attention_probs_dropout_prob, name: join(name, "dropout"))
      |> Layers.apply_layer_head_mask(layer_head_mask)

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()

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
    |> Layers.activation(config.hidden_act, name: join(name, "activation"))
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
    |> Layers.take_token(index: 0, axis: 1, name: join(name, "head"))
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
    |> lm_prediction_head_transform(config, name: name)
    # We reuse the kernel of input embeddings and add bias for each token
    |> Layers.dense_transposed(config.vocab_size,
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
    |> Layers.activation(config.hidden_act, name: join(name, "activation"))
    |> Axon.layer_norm(
      epsilon: config.layer_norm_eps,
      name: join(name, "layer_norm"),
      channel_index: 2
    )
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
