defmodule Bumblebee.Diffusion.PndmSchedulerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "timesteps" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler)

    assert_raise ArgumentError, "expected at least 4 steps, got: 2", fn ->
      scheduler_timesteps(scheduler, 2)
    end

    timesteps = scheduler_timesteps(scheduler, 4)

    assert_equal(
      timesteps,
      Nx.tensor([750, 625, 625, 500, 500, 375, 375, 250, 250, 125, 125, 0, 0])
    )

    timesteps = scheduler_timesteps(scheduler, 10)

    assert_equal(
      timesteps,
      Nx.tensor(
        [900, 850, 850, 800, 800, 750, 750, 700, 700, 650, 650, 600, 600, 500] ++
          [400, 300, 200, 100, 0]
      )
    )
  end

  test "timesteps with reduced warmup" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler, reduce_warmup: true)

    timesteps = scheduler_timesteps(scheduler, 2)
    assert_equal(timesteps, Nx.tensor([500, 0, 0]))

    timesteps = scheduler_timesteps(scheduler, 4)
    assert_equal(timesteps, Nx.tensor([750, 500, 500, 250, 0]))

    timesteps = scheduler_timesteps(scheduler, 10)
    assert_equal(timesteps, Nx.tensor([900, 800, 800, 700, 600, 500, 400, 300, 200, 100, 0]))
  end

  test "default configuration" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0747, 0.0767], [0.0827, 0.0848], [0.0908, 0.0928]],
        [[0.1393, 0.1413], [0.1473, 0.1493], [0.1554, 0.1574]],
        [[0.2038, 0.2058], [0.2119, 0.2139], [0.2200, 0.2220]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(65.8717))
  end

  test "with reduced warmup" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler, reduce_warmup: true)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0745, 0.0765], [0.0826, 0.0846], [0.0906, 0.0926]],
        [[0.1390, 0.1410], [0.1470, 0.1490], [0.1551, 0.1571]],
        [[0.2034, 0.2054], [0.2115, 0.2135], [0.2195, 0.2215]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(65.7333))
  end

  test ":quadratic beta schedule" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler, beta_schedule: :quadratic)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0742, 0.0762], [0.0822, 0.0842], [0.0902, 0.0922]],
        [[0.1384, 0.1404], [0.1464, 0.1484], [0.1544, 0.1564]],
        [[0.2025, 0.2045], [0.2105, 0.2125], [0.2186, 0.2206]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(65.4468))
  end

  test ":squared_cosine beta schedule" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler, beta_schedule: :squared_cosine)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0740, 0.0760], [0.0820, 0.0840], [0.0900, 0.0920]],
        [[0.1380, 0.1400], [0.1460, 0.1480], [0.1540, 0.1560]],
        [[0.2021, 0.2041], [0.2101, 0.2121], [0.2181, 0.2201]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(65.2970))
  end

  test "beta schedule range" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler, beta_start: 0.001, beta_end: 0.02)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0752, 0.0773], [0.0834, 0.0854], [0.0915, 0.0935]],
        [[0.1403, 0.1423], [0.1484, 0.1505], [0.1566, 0.1586]],
        [[0.2054, 0.2074], [0.2135, 0.2156], [0.2217, 0.2237]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(66.3751))
  end

  test ":angular_velocity prediction type" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.PndmScheduler, prediction_type: :angular_velocity)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0254, 0.0261], [0.0281, 0.0288], [0.0309, 0.0316]],
        [[0.0474, 0.0481], [0.0501, 0.0508], [0.0529, 0.0535]],
        [[0.0693, 0.0700], [0.0721, 0.0728], [0.0748, 0.0755]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(22.4076))
  end
end
