defmodule Bumblebee.Text.ClipText do
  @common_keys [
    :output_hidden_states,
    :output_attentions,
    :id2label,
    :label2id,
    :num_labels
  ]
  @moduledoc """
  Models based on CLIP architecture.
  """

  # TODO: Should this include the joint model as well?

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              vocab_size: 49408,
              hidden_size: 512,
              intermediate_size: 2048,
              num_hidden_layers: 12,
              num_attention_heads: 8,
              max_position_embeddings: 77,
              hidden_act: :quick_gelu,
              layer_norm_eps: 1.0e-5,
              dropout: 0.0,
              attention_dropout: 0.0,
              initializer_range: 0.02,
              initializer_factor: 1.0,
              # Tokens
              pad_token_id: 1,
              bos_token_id: 0,
              eos_token_id: 2
            ] ++ Shared.common_config_defaults(@common_keys)

  @behaviour Bumblebee.ModelSpec

  @impl true
  def architectures(), do: [:base]

  @impl true
  def base_model_prefix(), do: "text_model"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def input_template(_config) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = config) do
    inputs = inputs()

    inputs
    |> clip_text(config)
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

  defp clip_text(inputs, config, opts \\ []) do
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

    text_transformer(input_ids, attention_mask, position_ids, config, name: name)
  end

  defp text_transformer(input_ids, attention_mask, position_ids, config, opts) do
    name = opts[:name]

    hidden_state =
      text_embeddings(input_ids, position_ids, config, name: join(name, "embeddings"))

    encoder_outputs =
      encoder(hidden_state, attention_mask, config, name: join(name, "encoder"), causal?: true)

    last_hidden_state =
      Axon.layer_norm(encoder_outputs.last_hidden_state,
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "final_layer_norm")
      )

    pooler_output =
      Axon.layer(
        fn last_hidden_state, input_ids, _opts ->
          eos_idx = Nx.argmax(input_ids, axis: -1)
          Bumblebee.Utils.Nx.batched_take(last_hidden_state, eos_idx)
        end,
        [last_hidden_state, input_ids]
      )

    %{
      last_hidden_state: last_hidden_state,
      pooler_output: pooler_output,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp text_embeddings(input_ids, position_ids, config, opts) do
    name = opts[:name]

    input_embeds =
      Axon.embedding(input_ids, config.vocab_size, config.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "token_embedding")
      )

    position_embeds =
      Axon.embedding(position_ids, config.max_position_embeddings, config.hidden_size,
        kernel_initializer: Axon.Initializers.normal(),
        name: join(name, "position_embedding")
      )

    Axon.add(input_embeds, position_embeds)
  end

  defp encoder(input_embeds, attention_mask, config, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    encoder_layer_collection(input_embeds, attention_mask, config, name: name, causal?: causal?)
  end

  defp encoder_layer_collection(hidden_state, attention_mask, config, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    state = %{
      last_hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, config.output_hidden_states),
      attentions: Layers.maybe_container({}, config.output_attentions)
    }

    for idx <- 0..(config.num_hidden_layers - 1), reduce: state do
      state ->
        {hidden_state, attention} =
          encoder_layer(state.last_hidden_state, attention_mask, config,
            name: join(name, "layers.#{idx}"),
            causal?: causal?
          )

        %{
          last_hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          attentions: Layers.append(state.attentions, attention)
        }
    end
  end

  defp encoder_layer(hidden_state, attention_mask, config, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    residual = hidden_state

    {hidden_state, attention_weights} =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "layer_norm1")
      )
      |> attention(attention_mask, config, name: join(name, "self_attn"), causal?: causal?)

    hidden_state = Axon.add(residual, hidden_state)

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(
        channel_index: 2,
        epsilon: config.layer_norm_eps,
        name: join(name, "layer_norm2")
      )
      |> mlp(config, name: join(name, "mlp"))
      |> Axon.add(residual)

    {hidden_state, attention_weights}
  end

  defp attention(hidden_state, attention_mask, config, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    num_heads = config.num_attention_heads

    query =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "q_proj")
      )
      |> Layers.split_heads(num_heads)

    value =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "v_proj")
      )
      |> Layers.split_heads(num_heads)

    key =
      hidden_state
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
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
      |> Axon.dropout(rate: config.attention_dropout, name: join(name, "dropout"))

    attention_output =
      attention_weights
      |> Layers.attention_output(value)
      |> Layers.flatten_trailing()
      |> Axon.dense(config.hidden_size,
        kernel_initializer: kernel_initializer(config),
        name: join(name, "out_proj")
      )

    {attention_output, attention_weights}
  end

  defp mlp(hidden_state, config, opts) do
    name = opts[:name]

    hidden_state
    |> Axon.dense(config.intermediate_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "fc1")
    )
    |> Layers.activation(config.hidden_act, name: join(name, "activation"))
    |> Axon.dense(config.hidden_size,
      kernel_initializer: kernel_initializer(config),
      name: join(name, "fc2")
    )
  end

  defp kernel_initializer(_config) do
    Axon.Initializers.normal(scale: 0.01)
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
