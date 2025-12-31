defmodule Bumblebee.Text.GptBigCode do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50257,
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
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Transformer blocks in the decoder"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      num_key_value_heads: [
        default: nil,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      intermediate_size: [
        default: nil,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the decoder.
        If not specified, defaults to 4 times `:hidden_size`
        """
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      scale_attention_weights: [
        default: true,
        doc: "whether to scale attention weights to have variance of 1"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embedding and encoder"
      ],
      embeddings_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embeddings"
      ],
      attention_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for attention weights"
      ],
      classifier_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for the classification head"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([:use_cross_attention, :num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 50256)

  @moduledoc """
  GPT-BigCode model family.

  ## Architectures

    * `:base` - plain GPT-BigCode without any head on top

    * `:for_causal_language_modeling` - GPT-BigCode with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - GPT-BigCode with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - GPT-BigCode with a token classification
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

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"encoder_hidden_state"` - `{batch_size, encoder_sequence_length, hidden_size}`

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

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

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
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling,
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

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    # TODO: Tie lm-head to word embedding as a spec option
    logits =
      Layers.dense_transposed(outputs.hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: "language_modeling_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
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
        name: "token_classification_head.output"
      )

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
      Layers.dense_transposed(outputs.hidden_state, spec.num_labels,
        name: "sequence_classification_head.output"
      )

    pooled_logits =
      Layers.if_present inputs["input_ids"] do
        Axon.layer(
          fn logits, input_ids, _opts ->
            indices =
              input_ids
              |> Nx.not_equal(spec.pad_token_id)
              |> Nx.sum(axes: [-1])
              |> Nx.subtract(1)
              |> Nx.as_type({:s, 64})

            Bumblebee.Utils.Nx.batched_take(logits, indices)
          end,
          [logits, inputs["input_ids"]]
        )
      else
        Layers.take_token(logits, axis: 1, index: -1)
      end

    Layers.output(%{
      logits: pooled_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions
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

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}
    decoder_attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: decoder_attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
      Axon.input("encoder_attention_mask", optional: true, shape: shape),
      Axon.input("cross_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(inputs["input_ids"], inputs["position_ids"], inputs["input_embeddings"], spec,
        name: "embedder"
      )

    outputs =
      decoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["encoder_hidden_state"],
        inputs["encoder_attention_mask"],
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      Axon.layer_norm(outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: "norm"
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.replace(outputs.hidden_states, -1, hidden_state),
      attentions: outputs.attentions,
      cross_attentions: outputs.cross_attentions,
      cache: outputs.cache
    }
  end

  defp embedder(input_ids, position_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
          name: join(name, "token_embedding")
        )
      end

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_embeddings)
      end

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    input_embeddings
    |> Axon.add(position_embeddings)
    |> Axon.dropout(rate: spec.embeddings_dropout_rate)
  end

  defp decoder(
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

    Layers.Transformer.blocks(
      hidden_state,
      [
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        cache: cache,
        causal: true,
        num_blocks: spec.num_blocks,
        num_attention_heads: spec.num_attention_heads,
        num_key_value_heads: spec.num_key_value_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [
          epsilon: spec.layer_norm_epsilon
        ],
        ffn: [
          intermediate_size: spec.intermediate_size || 4 * spec.hidden_size,
          activation: spec.activation
        ],
        block_type: :norm_first,
        attention_scale: if(not spec.scale_attention_weights, do: 1),
        name: join(name, "blocks")
      ] ++
        if(spec.use_cross_attention,
          do: [
            cross_hidden_state: encoder_hidden_state,
            cross_attention_mask: encoder_attention_mask,
            cross_attention_head_mask: cross_attention_head_mask
          ],
          else: []
        )
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
          max_positions: {"n_positions", number()},
          hidden_size: {"n_embd", number()},
          num_blocks: {"n_layer", number()},
          num_attention_heads: {"n_head", number()},
          num_key_value_heads: {"multi_query", mapping(%{true => 1, false => nil})},
          intermediate_size: {"n_inner", optional(number())},
          activation: {"activation_function", activation()},
          scale_attention_weights: {"scale_attn_weights", boolean()},
          dropout_rate: {"resid_pdrop", number()},
          embeddings_dropout_rate: {"embd_pdrop", number()},
          attention_dropout_rate: {"attn_pdrop", number()},
          classifier_dropout_rate: {"classifier_dropout", number()},
          layer_norm_epsilon: {"layer_norm_epsilon", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      # The QKV parameters are joint, but they have different layout
      # depending on the "multi_query" option, so we slice accordingly
      out_template =
        if spec.num_key_value_heads == 1 do
          {[spec.num_attention_heads, spec.num_key_value_heads, spec.num_key_value_heads], :auto}
        else
          {spec.num_attention_heads, [1, 1, 1], :auto}
        end

      %{
        "embedder.token_embedding" => "transformer.wte",
        "embedder.position_embedding" => "transformer.wpe",
        "decoder.blocks.{n}.self_attention.query" =>
          Shared.sliced_dense_params_source("transformer.h.{n}.attn.c_attn", out_template, 0),
        "decoder.blocks.{n}.self_attention.key" =>
          Shared.sliced_dense_params_source("transformer.h.{n}.attn.c_attn", out_template, 1),
        "decoder.blocks.{n}.self_attention.value" =>
          Shared.sliced_dense_params_source("transformer.h.{n}.attn.c_attn", out_template, 2),
        "decoder.blocks.{n}.self_attention.output" => "transformer.h.{n}.attn.c_proj",
        "decoder.blocks.{n}.self_attention_norm" => "transformer.h.{n}.ln_1",
        "decoder.blocks.{n}.cross_attention.query" => "transformer.h.{n}.crossattention.q_attn",
        "decoder.blocks.{n}.cross_attention.key" =>
          Shared.sliced_dense_params_source(
            "transformer.h.{n}.crossattention.c_attn",
            {spec.num_attention_heads, [1, 1], :auto},
            0
          ),
        "decoder.blocks.{n}.cross_attention.value" =>
          Shared.sliced_dense_params_source(
            "transformer.h.{n}.crossattention.c_attn",
            {spec.num_attention_heads, [1, 1], :auto},
            1
          ),
        "decoder.blocks.{n}.cross_attention.output" => "transformer.h.{n}.crossattention.c_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "transformer.h.{n}.mlp.c_fc",
        "decoder.blocks.{n}.ffn.output" => "transformer.h.{n}.mlp.c_proj",
        "decoder.blocks.{n}.cross_attention_norm" => "transformer.h.{n}.ln_cross_attn",
        "decoder.blocks.{n}.output_norm" => "transformer.h.{n}.ln_2",
        "norm" => "transformer.ln_f",
        "language_modeling_head.output" => "transformer.wte",
        "token_classification_head.output" => "classifier",
        "sequence_classification_head.output" => "score"
      }
    end
  end
end
