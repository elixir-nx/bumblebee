defmodule Bumblebee.Audio.Whisper do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 51865,
        doc: """
        the vocabulary size of the model. This corresponds to the number of distinct
        tokens that can be represented by the decoder
        """
      ],
      feature_size: [
        default: 80,
        doc: """
        the dimensionality of the input features. This corresponds to the number of Mel
        bins in the preprocessed input
        """
      ],
      encoder_max_positions: [
        default: 1500,
        doc: """
        the vocabulary size of the encoder position embedding. This corresponds to the maximum
        sequence length of log-mel filter-bank features that the model can process
        """
      ],
      decoder_max_positions: [
        default: 448,
        doc: """
        the vocabulary size of the decoder position embedding. This corresponds to the maximum
        sequence length that this model can generate. Typically this is set to a large value just
        in case, such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      encoder_num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      decoder_num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the decoder"
      ],
      encoder_num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      decoder_num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      encoder_intermediate_size: [
        default: 4096,
        docs:
          "the dimensionality of the intermediate (often named feed-forward) layer in the encoder"
      ],
      decoder_intermediate_size: [
        default: 4096,
        docs:
          "the dimensionality of the intermediate (often named feed-forward) layer in the decoder"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for encoder and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      activation_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for activations inside fully connected layers"
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
  Whisper model family.

  ## Architectures

    * `:base` - plain Whisper without any head on top

    * `:for_conditional_generation` - Whisper with a language modeling
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_features"` - `{batch_size, input_length, feature_size}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_features"`, which can be specified
      for more control over how `"input_features"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_features"` will be ignored.

    * `"decoder_input_ids"` - `{batch_size, target_sequence_length}`

      Indices of decoder input sequence tokens in the vocabulary.

    * `"decoder_attention_mask"` - `{batch_size, target_sequence_length}`

      Mask indicating which decoder tokens to attend to. This is used
      to ignore padding tokens, which are added when processing a batch
      of sequences with different length.

    * `"decoder_attention_head_mask"` - `{decoder_num_blocks, decoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the decoder.

    * `"decoder_input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"decoder_input_ids"`, which can be
      specified for more control over how `"decoder_input_ids"` are
      embedded than the model's internal embedding lookup. If
      `"decoder_input_embeddings"` are present, then `"decoder_input_ids"`
      will be ignored.

    * `"encoder_hidden_state"` - `{batch_size, sequence_length, hidden_size}`

      Last hidden state output from the encoder. This hidden state is
      used in cross-attention blocks in the decoder. If specified, the
      model will skip the encoding process and use this value directly
      for cross-attentions in the decoder.

    * `"cross_attention_head_mask"` - `{decoder_num_blocks, decoder_num_attention_heads}`

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
  def architectures(), do: [:base, :for_conditional_generation]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
  end

  @impl true
  def input_template(spec) do
    input_length = 2 * spec.encoder_max_positions

    %{
      "input_features" => Nx.template({1, input_length, spec.feature_size}, :f32),
      "decoder_input_ids" => Nx.template({1, 1}, :u32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_conditional_generation} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      Layers.dense_transposed(outputs.hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: "language_modeling_head.output"
      )

    Layers.output(%{
      logits: logits,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_hidden_state: outputs.encoder_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions,
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
      decoder_num_attention_heads: spec.decoder_num_attention_heads,
      encoder_num_attention_heads: spec.encoder_num_attention_heads,
      decoder_num_blocks: spec.decoder_num_blocks,
      encoder_sequence_length: encoder_sequence_length
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  @impl true
  def extra_config_module(_spec), do: Bumblebee.Text.WhisperGenerationConfig

  defp inputs(spec) do
    input_length = 2 * spec.encoder_max_positions

    encoder_input_shape = {nil, input_length, spec.feature_size}
    decoder_input_shape = {nil, nil}

    encoder_attention_head_mask_shape =
      {spec.encoder_num_blocks, spec.encoder_num_attention_heads}

    decoder_attention_head_mask_shape =
      {spec.decoder_num_blocks, spec.decoder_num_attention_heads}

    hidden_shape = {nil, nil, spec.hidden_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_features", optional: true, shape: encoder_input_shape),
      Axon.input("attention_head_mask", optional: true, shape: encoder_attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("decoder_input_ids", optional: true, shape: decoder_input_shape),
      Axon.input("decoder_attention_mask", optional: true, shape: decoder_input_shape),
      Axon.input("decoder_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("decoder_input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("encoder_hidden_state", optional: true, shape: hidden_shape),
      Axon.input("cross_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    encoder_outputs =
      Layers.if_present inputs["encoder_hidden_state"] do
        %{
          hidden_state: inputs["encoder_hidden_state"],
          hidden_states: Layers.none(),
          attentions: Layers.none()
        }
      else
        embeddings =
          encoder_embedder(inputs["input_features"], inputs["input_embeddings"], spec,
            name: "encoder_embedder"
          )

        embeddings
        |> encoder(inputs["attention_head_mask"], spec, name: "encoder")
        |> Map.take([:hidden_state, :hidden_states, :attentions])
      end

    embeddings =
      decoder_embedder(
        inputs["decoder_input_ids"],
        inputs["decoder_input_embeddings"],
        inputs["cache"],
        spec,
        name: "decoder_embedder"
      )

    decoder_outputs =
      decoder(
        embeddings,
        inputs["decoder_attention_mask"],
        inputs["decoder_attention_head_mask"],
        encoder_outputs.hidden_state,
        inputs["cross_attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    %{
      hidden_state: decoder_outputs.hidden_state,
      decoder_hidden_states: decoder_outputs.hidden_states,
      decoder_attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions,
      cache: decoder_outputs.cache,
      encoder_hidden_state: encoder_outputs.hidden_state,
      encoder_hidden_states: encoder_outputs.hidden_states,
      encoder_attentions: encoder_outputs.attentions
    }
  end

  defp encoder_embedder(input_features, input_embedding, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embedding do
        feature_embedding(input_features, spec, name: join(name, "feature_embedding"))
      end

    position_embeddings =
      Layers.learned_embeddings(spec.encoder_max_positions, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    Axon.add([input_embeddings, position_embeddings])
    |> Axon.dropout(rate: spec.dropout_rate)
  end

  defp feature_embedding(input_features, spec, opts) do
    name = opts[:name]

    input_features
    |> Axon.conv(spec.hidden_size,
      kernel_size: 3,
      padding: [{1, 1}],
      name: join(name, "conv_1")
    )
    |> Axon.gelu()
    |> Axon.conv(spec.hidden_size,
      kernel_size: 3,
      strides: [2],
      padding: [{1, 1}],
      name: join(name, "conv_2")
    )
    |> Axon.gelu()
  end

  defp decoder_embedder(input_ids, input_embeddings, cache, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
          name: join(name, "token_embedding")
        )
      end

    position_embeddings =
      Layers.learned_embeddings(spec.decoder_max_positions, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    offset = Layers.Decoder.get_cache_offset(cache)

    # We use learned position embeddings for the maximum sequence
    # length, so to add them to the given input embeddings we need
    # to slice accordingly
    embeddings =
      Axon.layer(
        fn input_embeddings, position_embeddings, offset, _opts ->
          offset =
            case offset do
              %Axon.None{} -> 0
              offset -> offset
            end

          input_sequence_length = Nx.axis_size(input_embeddings, 1)

          position_embeddings =
            Nx.slice_along_axis(position_embeddings, offset, input_sequence_length, axis: 1)

          Nx.add(input_embeddings, position_embeddings)
        end,
        [input_embeddings, position_embeddings, Axon.optional(offset)]
      )

    embeddings
    |> Axon.dropout(rate: spec.dropout_rate)
  end

  defp encoder(hidden_state, attention_head_mask, spec, opts) do
    name = opts[:name]

    outputs =
      Layers.Transformer.blocks(hidden_state,
        attention_head_mask: attention_head_mask,
        num_blocks: spec.encoder_num_blocks,
        num_attention_heads: spec.encoder_num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        key_use_bias: false,
        layer_norm: [
          epsilon: 1.0e-5
        ],
        ffn: [
          intermediate_size: spec.encoder_intermediate_size,
          activation: spec.activation
        ],
        block_type: :norm_first,
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_attentions,
        name: join(name, "blocks")
      )

    hidden_state = Axon.layer_norm(outputs.hidden_state, name: join(name, "norm"))

    %{
      outputs
      | hidden_state: hidden_state,
        hidden_states: Layers.append(outputs.hidden_states, hidden_state)
    }
  end

  defp decoder(
         hidden_state,
         attention_mask,
         attention_head_mask,
         encoder_hidden_state,
         cross_attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]

    outputs =
      Layers.Transformer.blocks(hidden_state,
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        cross_hidden_state: encoder_hidden_state,
        cross_attention_head_mask: cross_attention_head_mask,
        cache: cache,
        causal: true,
        num_blocks: spec.decoder_num_blocks,
        num_attention_heads: spec.decoder_num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        key_use_bias: false,
        layer_norm: [
          epsilon: 1.0e-5
        ],
        ffn: [
          intermediate_size: spec.decoder_intermediate_size,
          activation: spec.activation
        ],
        block_type: :norm_first,
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_attentions,
        name: join(name, "blocks")
      )

    hidden_state = Axon.layer_norm(outputs.hidden_state, name: join(name, "norm"))

    %{
      outputs
      | hidden_state: hidden_state,
        hidden_states: Layers.append(outputs.hidden_states, hidden_state)
    }
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
          hidden_size: {"d_model", number()},
          feature_size: {"num_mel_bins", number()},
          encoder_max_positions: {"max_source_positions", number()},
          decoder_max_positions: {"max_target_positions", number()},
          encoder_num_blocks: {"encoder_layers", number()},
          decoder_num_blocks: {"decoder_layers", number()},
          encoder_num_attention_heads: {"encoder_attention_heads", number()},
          decoder_num_attention_heads: {"decoder_attention_heads", number()},
          encoder_intermediate_size: {"encoder_ffn_dim", number()},
          decoder_intermediate_size: {"decoder_ffn_dim", number()},
          activation: {"activation_function", activation()},
          dropout_rate: {"dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          activation_dropout_rate: {"activation_dropout", number()},
          initializer_scale: {"init_std", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "encoder_embedder.feature_embedding.conv_1" => "model.encoder.conv1",
        "encoder_embedder.feature_embedding.conv_2" => "model.encoder.conv2",
        "encoder_embedder.position_embedding" => %{
          "embeddings" => {[{"model.encoder.embed_positions", "weight"}], fn [value] -> value end}
        },
        "encoder.blocks.{n}.self_attention.query" => "model.encoder.layers.{n}.self_attn.q_proj",
        "encoder.blocks.{n}.self_attention.key" => "model.encoder.layers.{n}.self_attn.k_proj",
        "encoder.blocks.{n}.self_attention.value" => "model.encoder.layers.{n}.self_attn.v_proj",
        "encoder.blocks.{n}.self_attention.output" =>
          "model.encoder.layers.{n}.self_attn.out_proj",
        "encoder.blocks.{n}.self_attention_norm" =>
          "model.encoder.layers.{n}.self_attn_layer_norm",
        "encoder.blocks.{n}.ffn.intermediate" => "model.encoder.layers.{n}.fc1",
        "encoder.blocks.{n}.ffn.output" => "model.encoder.layers.{n}.fc2",
        "encoder.blocks.{n}.output_norm" => "model.encoder.layers.{n}.final_layer_norm",
        "encoder.norm" => "model.encoder.layer_norm",
        "decoder_embedder.token_embedding" => "model.decoder.embed_tokens",
        "decoder_embedder.position_embedding" => %{
          "embeddings" => {[{"model.decoder.embed_positions", "weight"}], fn [value] -> value end}
        },
        "decoder.blocks.{n}.self_attention.query" => "model.decoder.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.decoder.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.decoder.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" =>
          "model.decoder.layers.{n}.self_attn.out_proj",
        "decoder.blocks.{n}.self_attention_norm" =>
          "model.decoder.layers.{n}.self_attn_layer_norm",
        "decoder.blocks.{n}.cross_attention.query" =>
          "model.decoder.layers.{n}.encoder_attn.q_proj",
        "decoder.blocks.{n}.cross_attention.key" =>
          "model.decoder.layers.{n}.encoder_attn.k_proj",
        "decoder.blocks.{n}.cross_attention.value" =>
          "model.decoder.layers.{n}.encoder_attn.v_proj",
        "decoder.blocks.{n}.cross_attention.output" =>
          "model.decoder.layers.{n}.encoder_attn.out_proj",
        "decoder.blocks.{n}.cross_attention_norm" =>
          "model.decoder.layers.{n}.encoder_attn_layer_norm",
        "decoder.blocks.{n}.ffn.intermediate" => "model.decoder.layers.{n}.fc1",
        "decoder.blocks.{n}.ffn.output" => "model.decoder.layers.{n}.fc2",
        "decoder.blocks.{n}.output_norm" => "model.decoder.layers.{n}.final_layer_norm",
        "decoder.norm" => "model.decoder.layer_norm",
        "language_modeling_head.output" => "model.decoder.embed_tokens"
      }
    end
  end
end
