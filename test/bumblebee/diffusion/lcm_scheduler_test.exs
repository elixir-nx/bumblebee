defmodule Bumblebee.Diffusion.LcmSchedulerTest do
  use ExUnit.Case, async: true

  test "invalid inputs" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler)

    assert_raise ArgumentError,
                 "expected the number of steps to be less or equal to the number of training steps (1000), got: 1001",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 1001, {1, 32, 32, 4})
                 end

    assert_raise ArgumentError,
                 "expected the number of steps to be less or equal to num_original_steps * strength (50 * 0.5). Either reduce the number of steps orincrease the strength",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 50, {1, 32, 32, 4},
                     strength: 0.5
                   )
                 end

    assert_raise ArgumentError,
                 "expected the number of steps to be less or equal to the number of original steps (50), got: 51",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 51, {1, 32, 32, 4})
                 end
  end

  test "decreasing timesteps" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler)

    {_state, timesteps} = Bumblebee.Diffusion.LcmScheduler.init(scheduler, 4, {1, 32, 32, 4})

    decreasing? =
      Nx.to_list(timesteps)
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.all?(fn [t1, t2] -> t1 > t2 end)

    assert decreasing?
  end
end
