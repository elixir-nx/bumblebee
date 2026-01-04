defmodule Bumblebee.Text.NomicBert do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 30528,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 8192,
        doc: """
        the maximum sequence length that this model can process. Typically this is set to a large
        value just in case, such as 512, 1024 or 2048
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
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 1000,
        doc: "base for computing rotary embedding frequency"
      ],
      rotary_embedding_percentage: [
        default: 1.0,
        doc: "percentage of hidden size to use for rotary embeddings"
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
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  Nomic BERT model family.

  This is a variant of BERT that uses:
  - Rotary position embeddings (RoPE) instead of absolute position embeddings
  - SwiGLU activation in the feed-forward network
  - Post-normalization (like original BERT)
  - No biases in attention and feed-forward layers

  ## Architectures

    * `:base` - plain Nomic BERT without any head on top

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence token used when applying
      rotary position embeddings (RoPE).

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

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

  defp inputs(spec) do
    shape = {nil, nil}
    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("token_type_ids", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape)
    ])
  end

  defp core(inputs, spec) do
    token_type_ids =
      Layers.default inputs["token_type_ids"] do
        Layers.default_token_type_ids(inputs["input_ids"])
      end

    embeddings = embedder(inputs["input_ids"], token_type_ids, spec, name: "embedder")

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    encoder_outputs =
      encoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        spec,
        name: "encoder"
      )

    # Mean pooling over non-masked tokens
    pooled_state =
      Layers.if_present inputs["attention_mask"] do
        Axon.layer(
          fn hidden_state, attention_mask, _opts ->
            # Expand mask for broadcasting with hidden_size
            mask = Nx.new_axis(attention_mask, -1)
            # Mask out padding tokens
            masked = Nx.multiply(hidden_state, mask)
            # Sum and normalize by actual sequence length
            sum = Nx.sum(masked, axes: [1])
            count = Nx.sum(mask, axes: [1])
            Nx.divide(sum, Nx.max(count, 1.0e-9))
          end,
          [encoder_outputs.hidden_state, inputs["attention_mask"]]
        )
      else
        Axon.layer(
          fn hidden_state, _opts ->
            Nx.mean(hidden_state, axes: [1])
          end,
          [encoder_outputs.hidden_state]
        )
      end

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(input_ids, token_type_ids, spec, opts) do
    name = opts[:name]

    token_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )

    token_type_embeddings =
      Axon.embedding(token_type_ids, spec.max_token_types, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_type_embedding")
      )

    Axon.add([token_embeddings, token_type_embeddings])
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
  end

  defp encoder(hidden_state, position_ids, attention_mask, attention_head_mask, spec, opts) do
    name = opts[:name]

    # Nomic BERT uses postnorm (like standard BERT):
    # Each block:
    #   attn_output = attention(hidden_states)
    #   hidden_states = norm1(attn_output + hidden_states)
    #   ffn_output = ffn(hidden_states)
    #   hidden_states = norm2(ffn_output + hidden_states)

    initial_state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({})
    }

    final_state =
      Enum.reduce(0..(spec.num_blocks - 1), initial_state, fn idx, state ->
        block_name = join(name, "blocks.#{idx}")

        # Self-attention (no prenorm)
        {attention_output, attention_weights, _cache, _bias} =
          Layers.Transformer.multi_head_attention(
            state.hidden_state,
            state.hidden_state,
            state.hidden_state,
            attention_mask: attention_mask,
            attention_head_mask:
              Layers.if_present attention_head_mask do
                Axon.nx(attention_head_mask, & &1[idx])
              end,
            num_heads: spec.num_attention_heads,
            hidden_size: spec.hidden_size,
            kernel_initializer: kernel_initializer(spec),
            causal: false,
            query_use_bias: false,
            key_use_bias: false,
            value_use_bias: false,
            output_use_bias: false,
            rotary_embedding: [
              position_ids: position_ids,
              max_positions: spec.max_positions,
              base: spec.rotary_embedding_base,
              percentage: spec.rotary_embedding_percentage
            ],
            name: join(block_name, "self_attention")
          )

        # Residual + Norm after attention (postnorm)
        hidden_state = Axon.add(attention_output, state.hidden_state)

        hidden_state =
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(block_name, "self_attention_norm")
          )

        # FFN
        ffn_output =
          gated_ffn(hidden_state, spec.intermediate_size, spec.hidden_size,
            name: join(block_name, "ffn"),
            activation: spec.activation
          )

        # Residual + Norm after FFN (postnorm)
        hidden_state = Axon.add(ffn_output, hidden_state)

        hidden_state =
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(block_name, "output_norm")
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention_weights)
        }
      end)

    %{
      hidden_state: final_state.hidden_state,
      hidden_states: final_state.hidden_states,
      attentions: final_state.attentions
    }
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    # Nomic MLP: y = fc11(x) * activation(fc12(x)), then fc2
    # fc11 is "up", fc12 is "gate", fc2 is "down"
    up =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "up"),
        use_bias: false
      )

    gate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "gate"),
        use_bias: false
      )

    # Nomic applies activation to gate, not up: up * activation(gate)
    hidden_state = Axon.multiply(up, Axon.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "down"), use_bias: false)
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
          max_token_types: {"type_vocab_size", number()},
          hidden_size: {"n_embd", number()},
          num_blocks: {"n_layer", number()},
          num_attention_heads: {"n_head", number()},
          intermediate_size: {"n_inner", optional(number())},
          rotary_embedding_base: {"rotary_emb_base", number()},
          rotary_embedding_percentage: {"rotary_emb_fraction", optional(number())},
          layer_norm_epsilon: {"layer_norm_epsilon", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      # Nomic uses 4 * n_embd for intermediate size if not specified
      opts =
        if opts[:intermediate_size] do
          opts
        else
          hidden_size = opts[:hidden_size] || spec.hidden_size
          Keyword.put(opts, :intermediate_size, 4 * hidden_size)
        end

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "embeddings.word_embeddings",
        "embedder.token_type_embedding" => "embeddings.token_type_embeddings",
        "embedder.norm" => "emb_ln",
        "encoder.blocks.{n}.self_attention.query" => qkv_dense("encoder.layers.{n}.attn.Wqkv", 0),
        "encoder.blocks.{n}.self_attention.key" => qkv_dense("encoder.layers.{n}.attn.Wqkv", 1),
        "encoder.blocks.{n}.self_attention.value" => qkv_dense("encoder.layers.{n}.attn.Wqkv", 2),
        "encoder.blocks.{n}.self_attention.output" => "encoder.layers.{n}.attn.out_proj",
        "encoder.blocks.{n}.self_attention_norm" => "encoder.layers.{n}.norm1",
        "encoder.blocks.{n}.ffn.up" => "encoder.layers.{n}.mlp.fc11",
        "encoder.blocks.{n}.ffn.gate" => "encoder.layers.{n}.mlp.fc12",
        "encoder.blocks.{n}.ffn.down" => "encoder.layers.{n}.mlp.fc2",
        "encoder.blocks.{n}.output_norm" => "encoder.layers.{n}.norm2"
      }
    end

    defp qkv_dense(source_layer_name, chunk_idx) do
      # Wqkv is [3 * hidden_size, hidden_size] in PyTorch format
      # After slicing, transpose to get [hidden_size, hidden_size] for Axon
      %{
        "kernel" => {
          [{source_layer_name, "weight"}],
          fn [kernel] ->
            size = Nx.axis_size(kernel, 0)
            step = div(size, 3)

            kernel
            |> Nx.slice_along_axis(chunk_idx * step, step, axis: 0)
            |> Nx.transpose()
          end
        }
      }
    end
  end
end
