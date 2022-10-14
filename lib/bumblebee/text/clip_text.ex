defmodule Bumblebee.Text.ClipText do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 49408,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 77,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 512,
        doc: "the dimensionality of hidden layers"
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
        default: 2048,
        doc:
          "the dimensionality of the intermediate (often named feed-forward) layer in the encoder"
      ],
      activation: [
        default: :quick_gelu,
        doc: "the activation function"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ]) ++ Shared.token_options(pad_token_id: 1, bos_token_id: 0, eos_token_id: 2)

  @moduledoc """
  The CLIP model for text encoding.

  ## Architectures

    * `:base` - the base text model

  ## Inputs

    * `"input_ids"` - `{batch_size, seq_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, seq_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.


    * `"position_ids"` - `{batch_size, seq_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  # TODO: add ClipVision and joint Clip

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts \\ []) do
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
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs()

    inputs
    |> clip_text(spec)
    |> Layers.output()
  end

  defp inputs(opts \\ []) do
    shape = Keyword.get(opts, :shape, {nil, nil})

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape)
    ])
  end

  defp clip_text(inputs, spec, opts \\ []) do
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

    text_transformer(input_ids, attention_mask, position_ids, spec, name: name)
  end

  defp text_transformer(input_ids, attention_mask, position_ids, spec, opts) do
    name = opts[:name]

    hidden_state = text_embeddings(input_ids, position_ids, spec, name: join(name, "embeddings"))

    encoder_outputs =
      encoder(hidden_state, attention_mask, spec, name: join(name, "encoder"), causal?: true)

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "final_layer_norm")
      )

    pooler_output =
      Axon.layer(
        fn hidden_state, input_ids, _opts ->
          eos_idx = Nx.argmax(input_ids, axis: -1)
          Bumblebee.Utils.Nx.batched_take(hidden_state, eos_idx)
        end,
        [hidden_state, input_ids]
      )

    %{
      hidden_state: hidden_state,
      pooler_output: pooler_output,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp text_embeddings(input_ids, position_ids, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "token_embedding")
      )

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "position_embedding")
      )

    Axon.add(input_embeddings, position_embeddings)
  end

  defp encoder(input_embeddings, attention_mask, spec, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    encoder_blocks(input_embeddings, attention_mask, spec, name: name, causal?: causal?)
  end

  defp encoder_blocks(hidden_state, attention_mask, spec, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, spec.output_hidden_states),
      attentions: Layers.maybe_container({}, spec.output_attentions)
    }

    for idx <- 0..(spec.num_blocks - 1), reduce: state do
      state ->
        {hidden_state, attention} =
          encoder_block(state.hidden_state, attention_mask, spec,
            name: join(name, "layers.#{idx}"),
            causal?: causal?
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention)
        }
    end
  end

  defp encoder_block(hidden_state, attention_mask, spec, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    residual = hidden_state

    {hidden_state, attention_weights} =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "layer_norm1")
      )
      |> attention(attention_mask, spec, name: join(name, "self_attn"), causal?: causal?)

    hidden_state = Axon.add(residual, hidden_state)

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "layer_norm2")
      )
      |> mlp(spec, name: join(name, "mlp"))
      |> Axon.add(residual)

    {hidden_state, attention_weights}
  end

  defp attention(hidden_state, attention_mask, spec, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    num_heads = spec.num_attention_heads

    query =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "q_proj")
      )
      |> Layers.split_heads(num_heads)

    value =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "v_proj")
      )
      |> Layers.split_heads(num_heads)

    key =
      hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "k_proj")
      )
      |> Layers.split_heads(num_heads)

    attention_mask = Layers.expand_attention_mask(attention_mask)

    attention_mask =
      if causal? do
        Layers.Decoder.apply_causal_mask(attention_mask, query, Axon.constant(Nx.tensor(0)))
      else
        attention_mask
      end

    attention_bias = Layers.attention_bias(attention_mask)

    attention_weights =
      Layers.attention_weights(query, key, attention_bias)
      |> Axon.dropout(rate: spec.attention_dropout_rate, name: join(name, "dropout"))

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "out_proj")
      )

    {attention_output, attention_weights}
  end

  defp mlp(hidden_state, spec, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(spec.intermediate_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "fc1")
    )
    |> Layers.activation(spec.activation, name: join(name, "activation"))
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "fc2")
    )
  end

  defp kernel_initializer(_spec) do
    Axon.Initializers.normal(scale: 0.01)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    # Support loading from the entire Clip spec
    def load(%{"model_type" => "clip", "text_spec" => spec}, data) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", atom()},
          attention_dropout_rate: {"attention_dropout", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
