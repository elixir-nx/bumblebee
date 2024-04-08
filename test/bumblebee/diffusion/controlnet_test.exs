defmodule Bumblebee.Diffusion.ControlNetTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-controlnet"})

    assert %Bumblebee.Diffusion.ControlNet{architecture: :base} = spec

    inputs = %{
      "sample" => Nx.broadcast(0.5, {1, 32, 32, 4}),
      "timestep" => Nx.tensor(1),
      "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 32}),
      "conditioning" => Nx.broadcast(0.5, {1, 64, 64, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.mid_block_state) == {1, 16, 16, 64}

    assert_all_close(
      outputs.mid_block_state[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [[-0.2818, 1.6207, -0.7002], [0.2391, 1.1387, 0.9682], [-0.6386, 0.7026, -0.4218]],
          [[1.0681, 1.8418, -1.0586], [0.9387, 0.5971, 1.2284], [1.2914, 0.4060, -0.9559]],
          [[0.5841, 1.2935, 0.0081], [0.7306, 0.2915, 0.7736], [0.0875, 0.9619, 0.4108]]
        ]
      ])
    )

    assert tuple_size(outputs.down_block_states) == 6

    first_down_block_state = elem(outputs.down_block_states, 0)
    assert Nx.shape(first_down_block_state) == {1, 32, 32, 32}

    assert_all_close(
      first_down_block_state[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [[-0.1423, 0.2804, -0.0497], [-0.1425, 0.2798, -0.0485], [-0.1426, 0.2794, -0.0488]],
          [[-0.1419, 0.2810, -0.0493], [-0.1427, 0.2803, -0.0479], [-0.1427, 0.2800, -0.0486]],
          [[-0.1417, 0.2812, -0.0494], [-0.1427, 0.2807, -0.0480], [-0.1426, 0.2804, -0.0486]]
        ]
      ])
    )

    last_down_block_state = elem(outputs.down_block_states, 5)
    assert Nx.shape(last_down_block_state) == {1, 16, 16, 64}

    assert_all_close(
      last_down_block_state[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [[-1.1169, 0.8087, 0.1024], [0.4832, 0.0686, 1.0149], [-0.3314, 0.1486, 0.4445]],
          [[0.5770, 0.3195, -0.2008], [1.5692, -0.1771, 0.7669], [0.4908, 0.1258, 0.0694]],
          [[0.4694, -0.3723, 0.1505], [1.7356, -0.4214, 0.8929], [0.4702, 0.2400, 0.1213]]
        ]
      ])
    )
  end
end
