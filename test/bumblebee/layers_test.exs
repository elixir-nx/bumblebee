defmodule Bumblebee.LayersTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.Layers

  describe "rotary_embedding/6" do
    test "applies dynamic scaling when sequence length exceeds max_positions" do
      query = Axon.input("query", shape: {1, 4, 1, 4})
      key = Axon.input("key", shape: {1, 4, 1, 4})
      position_ids = Axon.input("position_ids", shape: {1, 4})
      attention_mask = Axon.input("attention_mask", shape: {1, 4})

      {query, _key} =
        Layers.rotary_embedding(query, key, position_ids, attention_mask, 4,
          max_positions: 2,
          base: 10_000,
          scaling_strategy: %{type: :dynamic, factor: 2.0}
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

      scale = 2.0 * 4 / 2 - (2.0 - 1.0)
      base = 10_000 * :math.pow(scale, 4 / (4 - 2))
      angle = 1 / :math.sqrt(base)

      assert_all_close(
        outputs.query[[0, 1, 0, 1]],
        Nx.tensor(:math.cos(angle) - :math.sin(angle))
      )
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
        attention_factor: 0.1 * :math.log(4.0) + 1.0
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
        Nx.tensor([[-0.3012, 0.9975], [-1.3254, 0.9950], [-1.1311, 0.9925]])
        |> Nx.multiply(0.1 * :math.log(4.0) + 1.0),
        atol: 1.0e-4
      )
    end
  end
end
