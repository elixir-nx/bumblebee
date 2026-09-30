defmodule Bumblebee.Layers.TransformerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.Layers
  alias Bumblebee.Layers.Transformer

  test "custom attention receives self- and cross-attention options" do
    hidden = Axon.input("hidden", shape: {nil, nil, 4})
    context = Axon.input("context", shape: {nil, nil, 4})

    opts = [
      num_blocks: 2,
      num_attention_heads: 2,
      hidden_size: 4,
      ffn: [intermediate_size: 8],
      cross_hidden_state: context,
      name: "blocks"
    ]

    reference = hidden |> Transformer.blocks(opts) |> model_outputs()
    owner = self()

    attention = fn query, key, value, opts ->
      send(owner, {:attention, opts[:name]})
      Transformer.multi_head_attention(query, key, value, opts)
    end

    model = hidden |> Transformer.blocks([attention: attention] ++ opts) |> model_outputs()

    for index <- 0..1, kind <- ["self_attention", "cross_attention"] do
      name = "blocks.#{index}.#{kind}"
      assert_received {:attention, ^name}
    end

    inputs = %{
      "hidden" => Nx.iota({1, 3, 4}, type: :f32),
      "context" => Nx.iota({1, 2, 4}, type: :f32)
    }

    {init, predict} = Axon.build(model)
    params = init.(inputs, Axon.ModelState.empty())
    actual = predict.(params, inputs)
    expected = Axon.predict(reference, params, inputs)

    assert_all_close(actual.hidden_state, expected.hidden_state)

    for {left, right} <-
          Enum.zip(Tuple.to_list(actual.attentions), Tuple.to_list(expected.attentions)) do
      assert_all_close(left, right)
    end

    for {left, right} <-
          Enum.zip(
            Tuple.to_list(actual.cross_attentions),
            Tuple.to_list(expected.cross_attentions)
          ) do
      assert_all_close(left, right)
    end
  end

  test "block output is collected and passed to the following block" do
    hidden = Axon.input("hidden", shape: {nil, nil, 2})

    outputs =
      Transformer.blocks(hidden,
        num_blocks: 2,
        num_attention_heads: 1,
        hidden_size: 2,
        attention: fn query, _key, _value, opts ->
          {query, Layers.none(), opts[:attention_cache], Layers.none()}
        end,
        ffn: fn hidden, _name -> Axon.nx(hidden, &Nx.multiply(&1, 0)) end,
        layer_norm: fn hidden, _name -> hidden end,
        block_output: fn hidden, index -> Axon.nx(hidden, &Nx.add(&1, index + 1)) end
      )

    model = Axon.container(Map.take(outputs, [:hidden_state, :hidden_states]))
    inputs = Nx.tensor([[[1.0, 2.0]]])
    {init, predict} = Axon.build(model)
    actual = predict.(init.(inputs, Axon.ModelState.empty()), inputs)

    assert_equal(elem(actual.hidden_states, 0), inputs)
    assert_equal(elem(actual.hidden_states, 1), Nx.tensor([[[3.0, 5.0]]]))
    assert_equal(elem(actual.hidden_states, 2), Nx.tensor([[[8.0, 12.0]]]))
    assert_equal(actual.hidden_state, Nx.tensor([[[8.0, 12.0]]]))
  end

  defp model_outputs(outputs) do
    Axon.container(Map.take(outputs, [:hidden_state, :attentions, :cross_attentions]))
  end
end
