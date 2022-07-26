defmodule Bumblebee.Text.Albert do
  @common_keys [:output_hidden_states, :output_attentions, :id2label, :label2id, :num_labels]

  @moduledoc """
  Models based on the ALBERT architecture.

  ## Architectures

    * `:base` - plain ALBERT without any head on top

    * `:for_masked_language_modeling` - ALBERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_causal_language_modeling` - ALBERT with a language modeling
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

  ### Exceptions

  The `:for_multiple_choice` model accepts groups of sequences, so the
  expected sequence shape is `{batch_size, num_choices, seq_length}`.

  ## Configuration

    * `:vocab_size` - vocabulary size of the model. Defines the number
      of distinct tokens that can be represented by the in model input
      and output. Defaults to `30000`

    * `:embedding_size` - dimensionality of vocab embeddings. Defaults
      to 128

    * `:hidden_size` - dimensionality of the encoder layers and the
      pooler layer. Defaults to `4096`

    * `:num_hidden_layers` - the number of hidden layers in the
      Transformer encoder. Defaults to `12`

    * `:num_hidden_groups` - the number of groups for hidden layers,
      parameters in the same group are shared. Defaults to `1`

    * `:num_attention_heads` - the number of attention heads for each
      attention layer in the Transformer encoder. Defaults to `12`

    * `:intermediate_size` - dimensionality of the "intermediate"
      (often named feed-forward) layer in the Transformer encoder.
      Defaults to `16384`

    * `:inner_group_num` - number of inner repetition of attention
      and ffn. Defaults to 1

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

    * `:classifier_dropout_prob` - the dropout ratio for the classification
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
              vocab_size: 30000,
              embedding_size: 128,
              hidden_size: 4096,
              num_hidden_layers: 12,
              num_hidden_groups: 1,
              num_attention_heads: 12,
              intermediate_size: 16384,
              inner_group_num: 1,
              hidden_act: :gelu,
              hidden_dropout_prob: 0.0,
              attention_probs_dropout_prob: 0.0,
              max_position_embeddings: 512,
              type_vocab_size: 2,
              initializer_range: 0.02,
              layer_norm_eps: 1.0e-12,
              classifier_dropout_prob: 0.1,
              position_embedding_type: :absolute,
              pad_token_id: 0,
              bos_token_id: 2,
              eos_token_id: 3
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

  @impl true
  def base_model_prefix(), do: "albert"

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
      :for_pre_training
    ]

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
    %{"input_ids" => Nx.template({1, 1}, :s64)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs()
    |> albert(config, name: "albert")
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_masked_language_modeling} = config) do
    outputs = albert(inputs(), config, name: "albert")

    logits = lm_prediction_head(outputs.last_hidden_state, config, name: "predictions")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = config) do
    outputs = albert(inputs(), config, name: "albert")

    logits =
      outputs.pooler_output
      |> Axon.dropout(rate: classifier_dropout_rate(config))
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

  def model(%__MODULE__{architecture: :for_multiple_choice} = config) do
    inputs = inputs({nil, nil, nil})

    group_inputs = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]

    flat_inputs =
      Enum.reduce(group_inputs, inputs, fn name, inputs ->
        Map.update!(inputs, name, &Layers.flatten_leading/1)
      end)

    outputs = albert(flat_inputs, config, name: "albert")

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

  def model(%__MODULE__{architecture: :for_token_classification} = config) do
    outputs = albert(inputs(), config, name: "albert")

    logits =
      outputs.last_hidden_state
      |> Axon.dropout(rate: classifier_dropout_rate(config))
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
    outputs = albert(inputs(), config, name: "albert")

    logits =
      Axon.dense(outputs.last_hidden_state, 2,
        kernel_initializer: kernel_initializer(config),
        name: "qa_outputs"
      )

    start_logits = Axon.nx(logits, & &1[[0..-1//1, 0..-1//1, 0]])
    end_logits = Axon.nx(logits, & &1[[0..-1//1, 0..-1//1, 1]])

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
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

  defp albert(inputs, config, opts) do
    name = opts[:name]

    input_ids = inputs["input_ids"]

    attention_mask =
      Layers.default inputs["attention_mask"] do
        Layers.default_attention_mask(input_ids)
      end

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(input_ids)
      end

    token_type_ids =
      Layers.default inputs["token_type_ids"] do
        Layers.default_token_type_ids(input_ids)
      end

    hidden_state =
      embeddings(input_ids, position_ids, token_type_ids, config, name: join(name, "embeddings"))

    {last_hidden_state, hidden_states, attentions} =
      encoder(hidden_state, attention_mask, config, name: join(name, "encoder"))

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
      Axon.embedding(input_ids, config.vocab_size, config.embedding_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "word_embeddings")
      )

    position_embeds =
      Axon.embedding(position_ids, config.max_position_embeddings, config.embedding_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "position_embeddings")
      )

    token_type_embeds =
      Axon.embedding(token_type_ids, config.type_vocab_size, config.embedding_size,
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

  defp encoder(hidden_state, attention_mask, config, opts) do
    name = opts[:name]

    hidden_state =
      Axon.dense(hidden_state, config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "embedding_hidden_mapping_in")
      )

    albert_layer_groups(hidden_state, attention_mask, config,
      name: join(name, "albert_layer_groups")
    )
  end

  defp albert_layer_groups(hidden_state, attention_mask, config, opts) do
    name = opts[:name]

    hidden_states = Layers.maybe_container({hidden_state}, config.output_hidden_states)
    attentions = Layers.maybe_container({}, config.output_attentions)

    for idx <- 0..(config.num_hidden_layers - 1),
        reduce: {hidden_state, hidden_states, attentions} do
      {hidden_state, hidden_states, attentions} ->
        group_idx = div(idx, div(config.num_hidden_layers, config.num_hidden_groups))

        albert_layers(hidden_state, attention_mask, hidden_states, attentions, config,
          name: name |> join(group_idx) |> join("albert_layers")
        )
    end
  end

  defp albert_layers(hidden_state, attention_mask, hidden_states, attentions, config, opts) do
    name = opts[:name]

    for idx <- 0..(config.inner_group_num - 1),
        reduce: {hidden_state, hidden_states, attentions} do
      {hidden_state, hidden_states, attentions} ->
        {hidden_state, attention} =
          albert_layer(hidden_state, attention_mask, config, name: join(name, idx))

        {
          hidden_state,
          Layers.append(hidden_states, hidden_state),
          Layers.append(attentions, attention)
        }
    end
  end

  defp albert_layer(hidden_state, attention_mask, config, opts) do
    name = opts[:name]

    {attention_output, attention_weights} =
      self_attention(hidden_state, attention_mask, config, name: join(name, "attention"))

    hidden_state =
      attention_output
      |> Axon.dense(config.intermediate_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "ffn")
      )
      |> Layers.activation(config.hidden_act, name: join(name, "ffn.activation"))
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "ffn_output")
      )
      |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "ffn_output.dropout"))
      |> Axon.add(attention_output, name: join(name, "ffn.residual"))
      |> Axon.layer_norm(
        epsilon: config.layer_norm_eps,
        name: join(name, "full_layer_layer_norm"),
        channel_index: 2
      )

    {hidden_state, attention_weights}
  end

  defp self_attention(hidden_state, attention_mask, config, opts) do
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

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()

    projected =
      attention_output
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "dense")
      )
      |> Axon.dropout(rate: config.hidden_dropout_prob, name: join(name, "dense.dropout"))
      |> Axon.add(hidden_state)
      |> Axon.layer_norm(
        epsilon: config.layer_norm_eps,
        name: join(name, "LayerNorm"),
        channel_index: 2
      )

    {projected, attention_weights}
  end

  defp pooler(hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(axis: 1)
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: name
    )
    |> Axon.tanh(name: join(name, "tanh"))
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
  end

  defp lm_prediction_head_transform(hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(config.embedding_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "dense")
    )
    |> Layers.activation(config.hidden_act, name: join(name, "activation"))
    |> Axon.layer_norm(
      epsilon: config.layer_norm_eps,
      name: join(name, "LayerNorm"),
      channel_index: 2
    )
  end

  defp classifier_dropout_rate(config) do
    config.classifier_dropout_prob || config.hidden_dropout_prob
  end

  defp kernel_initializer(config) do
    Axon.Initializers.normal(scale: config.initializer_range)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      data
      |> Shared.convert_to_atom(["position_embedding_type", "hidden_act"])
      |> Shared.convert_common()
      |> Shared.data_into_config(config, except: [:architecture])
    end
  end
end
