defmodule Bumblebee.Text.GptNeoX do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 32000,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
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
      num_blocks: [
        default: 32,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_percentage: [
        default: 0.25,
        doc: "percentage of hidden dimensions to allocate to rotary embeddings"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency"
      ],
      classifier_dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for the classification head"
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
      use_parallel_transformer_block: [
        default: true,
        doc:
          "whether to use the parallel formulation of the Transformer block, where attention and FFN is computed independently"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ]) ++ Shared.token_options(pad_token_id: nil)

  @moduledoc """
  GPT-NeoX model family.

  ## Architectures

    * `:base` - plain GPT-NeoX without any head on top

    * `:for_causal_language_modeling` - GPT-NeoX with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - GPT-NeoX with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - GPT-NeoX with a token classification
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
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
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
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dropout(
        rate: spec.classifier_dropout_rate,
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
      Axon.layer_norm(decoder_outputs.hidden_state,
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

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      layer_norm: [
        epsilon: spec.layer_norm_epsilon
      ],
      ffn: [
        intermediate_size: spec.intermediate_size
      ],
      block_type: if(spec.use_parallel_transformer_block, do: :parallel, else: :norm_first),
      causal: true,
      rotary_embedding: [
        position_ids: position_ids,
        percentage: spec.rotary_embedding_percentage,
        base: spec.rotary_embedding_base
      ],
      query_use_bias: true,
      key_use_bias: true,
      value_use_bias: true,
      output_use_bias: true,
      output_hidden_states: spec.output_hidden_states,
      output_attentions: spec.output_attentions,
      name: join(name, "blocks")
    )
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

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_positions", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          rotary_embedding_percentage: {"rotary_pct", number()},
          rotary_embedding_base: {"rotary_emb_base", number()},
          classifier_dropout_rate: {"classifier_dropout", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"init_std", number()},
          use_parallel_transformer_block: {"use_parallel_residual", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "gpt_neox.embed_in",
        "decoder.blocks.{n}.self_attention.query" =>
          Shared.sliced_dense_params_source(
            "gpt_neox.layers.{n}.attention.query_key_value",
            {spec.num_attention_heads, [1, 1, 1], :auto},
            0
          ),
        "decoder.blocks.{n}.self_attention.key" =>
          Shared.sliced_dense_params_source(
            "gpt_neox.layers.{n}.attention.query_key_value",
            {spec.num_attention_heads, [1, 1, 1], :auto},
            1
          ),
        "decoder.blocks.{n}.self_attention.value" =>
          Shared.sliced_dense_params_source(
            "gpt_neox.layers.{n}.attention.query_key_value",
            {spec.num_attention_heads, [1, 1, 1], :auto},
            2
          ),
        "decoder.blocks.{n}.self_attention.output" => "gpt_neox.layers.{n}.attention.dense",
        "decoder.blocks.{n}.self_attention_norm" => "gpt_neox.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.self_attention.rotary_embedding" =>
          "gpt_neox.layers.{n}.self_attn.rotary_emb",
        "decoder.blocks.{n}.ffn.intermediate" => "gpt_neox.layers.{n}.mlp.dense_h_to_4h",
        "decoder.blocks.{n}.ffn.output" => "gpt_neox.layers.{n}.mlp.dense_4h_to_h",
        "decoder.blocks.{n}.output_norm" => "gpt_neox.layers.{n}.post_attention_layernorm",
        "output_norm" => "gpt_neox.final_layer_norm",
        "language_modeling_head.output" => "embed_out",
        "sequence_classification_head.output" => "score",
        "token_classification_head.output" => "classifier"
      }
    end
  end
end
