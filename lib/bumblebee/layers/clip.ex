defmodule Bumblebee.Layers.Clip do
  @moduledoc false

  # Functionality shared by ClipText and ClipVision models.

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @doc """
  Adds CLIP encoder to the network.

  Returns a map with encoder outputs.
  """
  def encoder(embeddings, attention_mask, spec, opts) do
    name = opts[:name]
    causal? = Keyword.get(opts, :causal?, false)

    encoder_blocks(embeddings, attention_mask, spec, name: name, causal?: causal?)
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
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "layer_norm1"))
      |> attention(attention_mask, spec, name: join(name, "self_attn"), causal?: causal?)

    hidden_state = Axon.add(residual, hidden_state)

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "layer_norm2"))
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
end
