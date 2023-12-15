defmodule Bumblebee.Text.BlipText do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 30524,
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
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      encoder_hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers in the vision encoder"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 8,
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
        default: 0.0,
        doc: "the dropout rate for embedding and encoder"
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
        :output_attentions
      ])

  @moduledoc """
  The BLIP model for text encoding.

  ## Architectures

    * `:base` - the base text model

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

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"encoder_hidden_state"` - `{batch_size, encoder_sequence_length, encoder_hidden_size}`

      Last hidden state output from the encoder. This hidden state is
      used in cross-attention blocks in the decoder. If specified, the
      model will skip the encoding process and use this value directly
      for cross-attentions in the decoder.

    * `"encoder_attention_mask"` - `{batch_size, encoder_sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"cross_attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the cross-attention blocks in
      the decoder with shape.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_causal_language_modeling]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :u32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec, decoder?: true)
    outputs = core(inputs, spec, decoder?: true)
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
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
    shape = {nil, nil}
    decoder? = Keyword.get(opts, :decoder?, false)
    hidden_shape = {nil, nil, spec.hidden_size}
    encoder_hidden_shape = {nil, nil, spec.encoder_hidden_size}
    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", optional: true, shape: shape),
        Axon.input("attention_mask", optional: true, shape: shape),
        Axon.input("position_ids", optional: true, shape: shape),
        Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
        Axon.input("input_embeddings", optional: true, shape: hidden_shape)
      ])

    extra_decoder_inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("encoder_hidden_state", optional: true, shape: encoder_hidden_shape),
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
      embedder(inputs["input_ids"], inputs["position_ids"], inputs["input_embeddings"], spec,
        name: "embedder"
      )

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

  defp embedder(input_ids, position_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "token_embedding")
        )
      end

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_embeddings)
      end

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "position_embedding")
      )

    input_embeddings
    |> Axon.add(position_embeddings)
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate)
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

    cross_attention? = decoder?

    Layers.Transformer.blocks(
      hidden_state,
      [
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
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
        ffn: [
          intermediate_size: spec.intermediate_size,
          activation: spec.activation
        ],
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_attentions,
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

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    # Support loading from the entire Blip configuration
    def load(spec, %{"model_type" => "blip", "text_config" => data}) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          encoder_hidden_size: {"encoder_hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      prefix =
        case spec.architecture do
          :base -> "text_model."
          :for_causal_language_modeling -> "text_decoder.bert."
        end

      %{
        "embedder.token_embedding" => prefix <> "embeddings.word_embeddings",
        "embedder.position_embedding" => prefix <> "embeddings.position_embeddings",
        "embedder.token_type_embedding" => prefix <> "embeddings.token_type_embeddings",
        "embedder.norm" => prefix <> "embeddings.LayerNorm",
        "encoder.blocks.{n}.self_attention.query" =>
          prefix <> "encoder.layer.{n}.attention.self.query",
        "encoder.blocks.{n}.self_attention.key" =>
          prefix <> "encoder.layer.{n}.attention.self.key",
        "encoder.blocks.{n}.self_attention.value" =>
          prefix <> "encoder.layer.{n}.attention.self.value",
        "encoder.blocks.{n}.self_attention.output" =>
          prefix <> "encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.self_attention_norm" =>
          prefix <> "encoder.layer.{n}.attention.output.LayerNorm",
        "encoder.blocks.{n}.cross_attention.query" =>
          prefix <> "encoder.layer.{n}.crossattention.self.query",
        "encoder.blocks.{n}.cross_attention.key" =>
          prefix <> "encoder.layer.{n}.crossattention.self.key",
        "encoder.blocks.{n}.cross_attention.value" =>
          prefix <> "encoder.layer.{n}.crossattention.self.value",
        "encoder.blocks.{n}.cross_attention.output" =>
          prefix <> "encoder.layer.{n}.crossattention.output.dense",
        "encoder.blocks.{n}.cross_attention_norm" =>
          prefix <> "encoder.layer.{n}.crossattention.output.LayerNorm",
        "encoder.blocks.{n}.ffn.intermediate" => prefix <> "encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => prefix <> "encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => prefix <> "encoder.layer.{n}.output.LayerNorm",
        "pooler.output" => prefix <> "pooler.dense",
        "language_modeling_head.dense" => "text_decoder.cls.predictions.transform.dense",
        "language_modeling_head.norm" => "text_decoder.cls.predictions.transform.LayerNorm",
        "language_modeling_head.output" => "text_decoder.cls.predictions.decoder",
        "language_modeling_head.bias" => "text_decoder.cls.predictions"
      }
    end
  end
end
