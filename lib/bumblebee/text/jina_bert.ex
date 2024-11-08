defmodule Bumblebee.Text.JinaBert do
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
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
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
      layer_norm_epsilon: [
        default: 1.0e-12,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++ Shared.common_options([:use_cross_attention, :num_labels, :id_to_label])

  @moduledoc """
  Jina adaption of BERT model family.

  ## Architectures

    * `:base` - plain Jina BERT without any head on top

    * `:for_masked_language_modeling` - Jina BERT with a language modeling
      head. The head returns logits for each token in the original
      sequence

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

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    * [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_masked_language_modeling
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

  @impl true
  def init_cache(spec, batch_size, max_length, inputs) do
    encoder_sequence_length =
      if encoder_hidden_state = inputs["encoder_hidden_state"] do
        Nx.axis_size(encoder_hidden_state, 1)
      end

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      encoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  defp inputs(spec, opts \\ []) do
    shape = Keyword.get(opts, :shape, {nil, nil})
    decoder? = Keyword.get(opts, :decoder?, false)

    hidden_shape = Tuple.append(shape, spec.hidden_size)
    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", shape: shape),
        Axon.input("attention_mask", optional: true, shape: shape),
        Axon.input("token_type_ids", optional: true, shape: shape),
        Axon.input("position_ids", optional: true, shape: shape),
        Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape)
      ])

    extra_decoder_inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
        Axon.input("encoder_attention_mask", optional: true, shape: shape),
        Axon.input("cross_attention_head_mask", optional: true, shape: attention_head_mask_shape),
        Axon.input("cache", optional: true)
      ])

    extra_decoder_inputs =
      if decoder? do
        extra_decoder_inputs
      else
        Map.new(extra_decoder_inputs, fn {name, _input} -> {name, Layers.none()} end)
      end

    Map.merge(inputs, extra_decoder_inputs)
  end

  defp core(inputs, spec, opts \\ []) do
    decoder? = Keyword.get(opts, :decoder?, false)

    embeddings =
      embedder(inputs["input_ids"], inputs["token_type_ids"], spec, name: "embedder")

    encoder_outputs =
      encoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["encoder_hidden_state"],
        inputs["encoder_attention_mask"],
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        decoder?: decoder?,
        name: "encoder"
      )

    pooled_state = pooler(encoder_outputs.hidden_state, spec, name: "pooler")

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      cross_attentions: encoder_outputs.cross_attentions,
      cache: encoder_outputs.cache
    }
  end

  defp embedder(input_ids, token_type_ids, spec, opts) do
    name = opts[:name]

    token_type_ids =
      Layers.default token_type_ids do
        Layers.default_token_type_ids(input_ids)
      end

    inputs_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )

    token_type_embeddings =
      Axon.embedding(token_type_ids, spec.max_token_types, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_type_embedding")
      )

    Axon.add([inputs_embeddings, token_type_embeddings])
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp get_slopes_power_of_2(n) do
    start = 2 ** -(2 ** -(:math.log2(n) - 3))
    ratio = start
    for i <- 0..(n - 1), do: start * ratio ** i
  end

  defp integer?(number) do
    round(number) == number
  end

  defp get_alibi_head_slopes(n_heads) do
    if integer?(:math.log2(n_heads)) do
      get_slopes_power_of_2(n_heads)
    else
      closest_power_of_2 = 2 ** round(:math.floor(:math.log2(n_heads)))

      get_slopes_power_of_2(closest_power_of_2) ++
        (get_alibi_head_slopes(2 * closest_power_of_2)
         |> Enum.take_every(2)
         |> Enum.take(n_heads - closest_power_of_2))
    end
  end

  defp alibi_matrix(num_attention_heads, size) do
    context_position = Nx.iota({1, size, 1}, axis: 1)
    memory_position = Nx.iota({1, size}, axis: 1)
    relative_position = Nx.abs(Nx.subtract(context_position, memory_position))

    relative_position = Nx.tile(relative_position, [num_attention_heads, 1, 1])
    slopes = Nx.tensor(get_alibi_head_slopes(num_attention_heads)) |> Nx.multiply(-1)

    slopes
    |> Nx.new_axis(-1)
    |> Nx.new_axis(-1)
    |> Nx.multiply(relative_position)
    |> Nx.new_axis(0)
  end

  defp encoder(
         hidden_state,
         attention_mask,
         attention_head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]
    decoder? = opts[:decoder?]

    cross_attention? = decoder? and spec.use_cross_attention

    alibi_relative_bias_matrix =
      Axon.nx(hidden_state, fn hidden_state ->
        {_, seqlen, _} = Nx.shape(hidden_state)

        matrix = alibi_matrix(spec.num_attention_heads, spec.max_positions)

        matrix[[.., .., 0..(seqlen - 1), 0..(seqlen - 1)]]
      end)

    Layers.Transformer.blocks(
      hidden_state,
      [
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        attention_relative_bias: alibi_relative_bias_matrix,
        cache: cache,
        causal: decoder?,
        num_blocks: spec.num_blocks,
        num_attention_heads: spec.num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [
          epsilon: spec.layer_norm_epsilon
        ],
        ffn: &glumlp(&1, spec, name: &2),
        block_type: &jina_block_impl/3,
        name: join(name, "blocks")
      ] ++
        if(cross_attention?,
          do: [
            cross_hidden_state: encoder_hidden_state,
            cross_attention_mask: encoder_attention_mask,
            cross_attention_head_mask: cross_attention_head_mask
          ],
          else: []
        )
    )
  end

  def glumlp(
        hidden_states,
        spec,
        opts
      ) do
    name = opts[:name]

    residual_connection = hidden_states

    hidden_states =
      Axon.dense(hidden_states, spec.intermediate_size * 2,
        use_bias: false,
        name: join(name, "gated_layers")
      )

    gated =
      Axon.nx(hidden_states, fn hidden_states ->
        hidden_states[[.., .., 0..(spec.intermediate_size - 1)]]
      end)
      |> Axon.activation(spec.activation)

    non_gated =
      Axon.nx(hidden_states, fn hidden_states ->
        hidden_states[[.., .., spec.intermediate_size..-1//1]]
      end)

    hidden_states =
      Axon.multiply(gated, non_gated)
      |> Axon.dropout(rate: spec.dropout_rate)
      |> Axon.dense(spec.hidden_size, name: join(name, "wo"))

    hidden_states
    |> Axon.add(residual_connection)
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "layernorm"))
  end

  defp pooler(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Layers.take_token(index: 0, axis: 1)
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

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp jina_block_impl(hidden_state, steps, _name) do
    shortcut = hidden_state

    {hidden_state, attention_info} = steps.self_attention.(hidden_state)

    hidden_state =
      hidden_state
      |> Axon.add(shortcut)
      |> steps.self_attention_norm.()

    {hidden_state, cross_attention_info} =
      steps.cross_attention_maybe.(hidden_state, fn hidden_state ->
        shortcut = hidden_state

        {hidden_state, cross_attention_info} = steps.cross_attention.(hidden_state)

        hidden_state =
          hidden_state
          |> Axon.add(shortcut)
          |> steps.cross_attention_norm.()

        {hidden_state, cross_attention_info}
      end)

    hidden_state = steps.ffn.(hidden_state)

    {hidden_state, attention_info, cross_attention_info}
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
        "embedder.token_embedding" => "bert.embeddings.word_embeddings",
        "embedder.position_embedding" => "bert.embeddings.position_embeddings",
        "embedder.token_type_embedding" => "bert.embeddings.token_type_embeddings",
        "embedder.norm" => "bert.embeddings.LayerNorm",
        "encoder.blocks.{n}.self_attention.query" =>
          "bert.encoder.layer.{n}.attention.self.query",
        "encoder.blocks.{n}.self_attention.key" => "bert.encoder.layer.{n}.attention.self.key",
        "encoder.blocks.{n}.self_attention.value" =>
          "bert.encoder.layer.{n}.attention.self.value",
        "encoder.blocks.{n}.self_attention.output" =>
          "bert.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.self_attention_norm" =>
          "bert.encoder.layer.{n}.attention.output.LayerNorm",
        "encoder.blocks.{n}.cross_attention.query" =>
          "bert.encoder.layer.{n}.crossattention.self.query",
        "encoder.blocks.{n}.cross_attention.key" =>
          "bert.encoder.layer.{n}.crossattention.self.key",
        "encoder.blocks.{n}.cross_attention.value" =>
          "bert.encoder.layer.{n}.crossattention.self.value",
        "encoder.blocks.{n}.cross_attention.output" =>
          "bert.encoder.layer.{n}.crossattention.output.dense",
        "encoder.blocks.{n}.cross_attention_norm" =>
          "bert.encoder.layer.{n}.crossattention.output.LayerNorm",
        "encoder.blocks.{n}.ffn.intermediate" => "bert.encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => "bert.encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => "bert.encoder.layer.{n}.output.LayerNorm",
        "pooler.output" => "bert.pooler.dense",
        "language_modeling_head.dense" => "cls.predictions.transform.dense",
        "language_modeling_head.norm" => "cls.predictions.transform.LayerNorm",
        "language_modeling_head.output" => "cls.predictions.decoder",
        "language_modeling_head.bias" => "cls.predictions",
        "next_sentence_prediction_head.output" => "cls.seq_relationship",
        "sequence_classification_head.output" => "classifier",
        "token_classification_head.output" => "classifier",
        "multiple_choice_head.output" => "classifier",
        "question_answering_head.output" => "qa_outputs",
        "encoder.blocks.{n}.ffn.wo" => "encoder.layer.{n}.mlp.wo",
        "encoder.blocks.{n}.ffn.layernorm" => "encoder.layer.{n}.mlp.layernorm",
        "encoder.blocks.{n}.ffn.gated_layers" => "encoder.layer.{n}.mlp.gated_layers"
      }
    end
  end
end
