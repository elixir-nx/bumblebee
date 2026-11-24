defmodule Bumblebee.LayersTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.Layers
  alias Bumblebee.Layers.Transformer

  describe "rotary embedding" do
    test "applies proportional scaling to a query at an explicit position" do
      query = Axon.input("query", shape: {1, 1, 1, 8})
      position_ids = Axon.input("position_ids", shape: {1, 1})

      model =
        Transformer.rotary_embedding(
          query,
          Layers.none(),
          8,
          [
            position_ids: position_ids,
            scaling_strategy: %{type: :proportional, percentage: 0.5, factor: 2.0}
          ],
          []
        )

      inputs = %{
        "query" => Nx.iota({1, 1, 1, 8}, type: :f32) |> Nx.add(1),
        "position_ids" => Nx.tensor([[5]])
      }

      outputs = Axon.predict(model, Axon.ModelState.empty(), inputs)

      assert_all_close(
        outputs,
        Nx.tensor([[[[-3.793504, 0.453401, 3.0, 4.0, -3.407246, 6.308282, 7.0, 8.0]]]]),
        atol: 1.0e-6
      )
    end

    test "changes dynamic scaling across the position limit with a fixed mask width" do
      query = Axon.input("query", shape: {1, 1, 1, 4})
      key = Axon.input("key", shape: {1, 1, 1, 4})
      position_ids = Axon.input("position_ids", shape: {1, 1})
      attention_mask = Axon.input("attention_mask", shape: {1, 8})

      {query, _key} =
        Layers.rotary_embedding(query, key, position_ids, attention_mask, 4,
          max_positions: 3,
          base: 10_000,
          scaling_strategy: %{type: :dynamic, factor: 2.0}
        )

      model = Axon.container(%{query: query})

      inputs = %{
        "query" => Nx.broadcast(1.0, {1, 1, 1, 4}),
        "key" => Nx.broadcast(1.0, {1, 1, 1, 4}),
        "position_ids" => Nx.tensor([[2]]),
        "attention_mask" => Nx.tensor([[0, 1, 1, 1, 0, 0, 0, 0]])
      }

      {init, predict} = Axon.build(model)
      params = init.(inputs, Axon.ModelState.empty())
      outputs = predict.(params, inputs)
      assert_all_close(outputs.query[[0, 0, 0, 1]], Nx.tensor(0.979801), atol: 1.0e-6)

      inputs = %{
        inputs
        | "position_ids" => Nx.tensor([[3]]),
          "attention_mask" => Nx.tensor([[0, 1, 1, 1, 1, 0, 0, 0]])
      }

      outputs = predict.(params, inputs)
      assert_all_close(outputs.query[[0, 0, 0, 1]], Nx.tensor(0.981839), atol: 1.0e-6)
    end

    test "applies YaRN scaling" do
      query = Axon.input("query", shape: {1, 4, 1, 4})
      key = Axon.input("key", shape: {1, 4, 1, 4})
      position_ids = Axon.input("position_ids", shape: {1, 4})
      attention_mask = Axon.input("attention_mask", shape: {1, 4})

      strategy = %{
        type: :yarn,
        factor: 4.0,
        original_max_positions: 2,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: 0.1 * :math.log(4.0) + 1.0,
        truncate: true
      }

      {query, _key} =
        Layers.rotary_embedding(query, key, position_ids, attention_mask, 4,
          scaling_strategy: strategy
        )

      model = Axon.container(%{query: query})

      inputs = %{
        "query" => Nx.broadcast(1.0, {1, 4, 1, 4}),
        "key" => Nx.broadcast(1.0, {1, 4, 1, 4}),
        "position_ids" => Nx.tensor([[0, 1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1]])
      }

      {init, predict} = Axon.build(model)
      params = init.(inputs, Axon.ModelState.empty())
      outputs = predict.(params, inputs)

      assert_all_close(
        outputs.query[[0, 1..3, 0, 0..1]],
        Nx.tensor([[-0.3429, 1.1358], [-1.5092, 1.1329], [-1.2879, 1.1301]])
      )
    end
  end
end
