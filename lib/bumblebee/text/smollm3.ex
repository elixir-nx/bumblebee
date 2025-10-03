defmodule Bumblebee.Text.SmolLM3 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 128_256,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 65536,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048.
        SmolLM3 supports up to 128k tokens with YaRN extrapolation.
        """
      ],
      hidden_size: [
        default: 4096,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 11008,
        doc: "the dimensionality of intermediate layers"
      ],
      attention_head_size: [
        default: nil,
        doc: """
        the size of the key, value, and query projection per attention head.
        Defaults to `div(hidden_size, num_attention_heads)`
        """
      ],
      num_blocks: [
        default: 32,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 4,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 5_000_000,
        doc: "base for computing rotary embedding frequency"
      ],
      rotary_embedding_scaling_strategy: [
        default: nil,
        doc: """
        scaling configuration for rotary embedding. Currently the supported values are:

          * `%{type: :linear, factor: number()}`

          * `%{type: :dynamic, factor: number()}`

          * `%{type: :llama3, factor: number(), low_frequency_factor: number(), high_frequency_factor: number(), original_max_positions: pos_integer()}`

          * `%{type: :yarn, factor: number(), original_max_positions: pos_integer()}`

        For more details see https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
        """
      ],
      no_rope_layers: [
        default: nil,
        doc:
          "a list containing 0 or 1 at the corresponding index for each layer. 0 means no rope layer, 1 means rope layer."
      ],
      layer_norm_epsilon: [
        default: 1.0e-12,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      tie_word_embeddings: [
        default: true,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++ Shared.token_options(pad_token_id: 0)

  @moduledoc """
  SmolLM3 is a 3B parameter language model designed to push the boundaries of small models.
  It supports dual mode reasoning, 6 languages and long context. SmolLM3 is a fully open model
  that offers strong performance at the 3B–4B scale.

  Key features

    * Instruct model optimized for hybrid reasoning
    * Fully open model: open weights + full training details including public data mixture and training configs
    * Long context: Trained on 64k context and supports up to 128k tokens using YARN extrapolation
    * Multilingual: 6 natively supported (English, French, Spanish, German, Italian, and Portuguese)

  For more details see: https://huggingface.co/HuggingFaceTB/SmolLM3-3B

  ## Architectures

    * `:base` - plain SmolLM3 without any head on top

    * `:for_causal_language_modeling` - SmolLM3 with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - SmolLM3 with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - SmolLM3 with a token classification
      head. The head returns logits for each token in the original
      sequence

    * `:for_question_answering` - SmolLM3 with a span classification head.
      The head returns logits for the span start and end positions

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
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
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
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output",
        use_bias: false
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
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dropout(
        rate: 0.1,
        name: "token_classification_head.dropout"
      )
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "token_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
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

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(
        inputs["input_ids"],
        inputs["input_embeddings"],
        spec,
        name: "embedder"
      )

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    Layers.default input_embeddings do
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
  end

  defp decoder(
         hidden_state,
         position_ids,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]

    rotary_embedding_config = [
      position_ids: position_ids,
      max_positions: spec.max_positions,
      base: spec.rotary_embedding_base,
      scaling_strategy: spec.rotary_embedding_scaling_strategy
    ]

    nope_rotary_embedding =
      case opts[:no_rope_layers] do
        nil ->
          rotary_embedding_config

        no_rope_layers ->
          fn layer_index ->
            if Enum.at(no_rope_layers, layer_index) == 1 do
              rotary_embedding_config
            else
              nil
            end
          end
      end

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      attention_head_size: spec.attention_head_size,
      cache: cache,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      layer_norm: &Layers.rms_norm(&1, name: &2, epsilon: spec.layer_norm_epsilon),
      ffn:
        &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
          name: &2,
          activation: spec.activation
        ),
      block_type: :norm_first,
      causal: true,
      rotary_embedding: nope_rotary_embedding,
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      output_use_bias: false,
      name: join(name, "blocks")
    )
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    gate = Axon.dense(hidden_state, intermediate_size, name: join(name, "gate"), use_bias: false)

    hidden_state = Axon.multiply(intermediate, Axon.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    # TODO: Tie lm-head to word embedding as a spec option
    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      scaling_strategy_converter = fn name, value ->
        # "type" has been renamed to "rope_type"
        value =
          case Map.pop(value, "type") do
            {nil, value} -> value
            {type, value} -> Map.put(value, "rope_type", type)
          end

        case value do
          %{"rope_type" => "linear", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :linear, factor: factor}}

          %{"rope_type" => "dynamic", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :dynamic, factor: factor}}

          %{
            "rope_type" => "llama3",
            "factor" => factor,
            "low_freq_factor" => low_frequency_factor,
            "high_freq_factor" => high_frequency_factor,
            "original_max_position_embeddings" => original_max_positions
          }
          when is_number(factor) and is_number(low_frequency_factor) and
                 is_number(high_frequency_factor) and
                 is_number(original_max_positions) ->
            {:ok,
             %{
               type: :llama3,
               factor: factor,
               low_frequency_factor: low_frequency_factor,
               high_frequency_factor: high_frequency_factor,
               original_max_positions: original_max_positions
             }}

          # TODO: implement yarn or find out if it's same as longrope
          %{
            "rope_type" => "yarn",
            "factor" => factor,
            "original_max_position_embeddings" => original_max_positions
          }
          when is_number(factor) and is_number(original_max_positions) ->
            {:ok,
             %{
               type: :yarn,
               factor: factor,
               original_max_positions: original_max_positions
             }}

          _other ->
            {:error, "invalid format for #{inspect(name)}, got: #{inspect(value)}"}
        end
      end

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          rotary_embedding_base: {"rope_theta", number()},
          rotary_embedding_scaling_strategy:
            {"rope_scaling", optional(scaling_strategy_converter)},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      base_mapping = %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.post_attention_layernorm",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head"),
        "sequence_classification_head.output" => "score",
        "token_classification_head.output" => "score",
        "question_answering_head.output" => "qa_outputs"
      }

      rotary_mapping =
        case spec.no_rope_layers do
          nil ->
            []

          no_rope_layers ->
            Enum.with_index(no_rope_layers, fn rope, index ->
              if rope == 1 do
                {"decoder.blocks.#{index}.self_attention.rotary_embedding",
                 "model.layers.#{index}.self_attn.rotary_emb"}
              end
            end)
        end

      mapping = Map.merge(base_mapping, Map.new(rotary_mapping))

      case spec do
        %{architecture: :for_question_answering} ->
          question_answering_mapping = %{
            "output_norm" => "transformer.norm",
            "embedder.token_embedding" => "transformer.embed_tokens",
            "decoder.blocks.0.output_norm" => "transformer.layers.0.post_attention_layernorm",
            "decoder.blocks.0.self_attention.key" => "transformer.layers.0.self_attn.k_proj",
            "decoder.blocks.0.self_attention.query" => "transformer.layers.0.self_attn.q_proj",
            "decoder.blocks.0.self_attention.value" => "transformer.layers.0.self_attn.v_proj",
            "decoder.blocks.0.self_attention_norm" => "transformer.layers.0.input_layernorm",
            "decoder.blocks.0.self_attention.output" => "transformer.layers.0.self_attn.o_proj",
            "decoder.blocks.0.ffn.output" => "transformer.layers.0.mlp.down_proj",
            "decoder.blocks.0.ffn.intermediate" => "transformer.layers.0.mlp.up_proj",
            "decoder.blocks.0.ffn.gate" => "transformer.layers.0.mlp.gate_proj"
          }

          Map.merge(mapping, question_answering_mapping)

        _else ->
          mapping
      end
    end
  end
end
