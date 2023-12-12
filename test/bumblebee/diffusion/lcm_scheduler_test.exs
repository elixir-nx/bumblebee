defmodule Bumblebee.Diffusion.LcmSchedulerTest do
  use ExUnit.Case, async: true

  test "invalid inputs" do
    scheduler = %Bumblebee.Diffusion.LcmScheduler{}

    assert_raise ArgumentError,
                 "the number of inference steps needs to be less than the original training timesteps (1000), got: 1001",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 1001, nil)
                 end

    assert_raise ArgumentError,
                 "the steps between lcm_origin_timesteps needs to be a positive integer, change inference steps (50) to be less than original_inference_steps (50) * strength (0.5)",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 50, nil, strength: 0.5)
                 end

    assert_raise ArgumentError,
                 "the number of inference steps (51) cannot be greater than original_inference_steps (50)",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 51, nil, strength: 2.0)
                 end
  end

  test "decreasing timesteps" do
    scheduler = %Bumblebee.Diffusion.LcmScheduler{}

    {_state, timesteps} = Bumblebee.Diffusion.LcmScheduler.init(scheduler, 4, nil)

    is_decreasing =
      Nx.to_list(timesteps)
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.all?(fn [t1, t2] -> t1 > t2 end)

    assert is_decreasing == true
  end
end
