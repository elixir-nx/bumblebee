defmodule Bumblebee.Diffusion.LcmSchedulerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "invalid inputs" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler)

    sample_template = Nx.template({1, 32, 32, 4}, :f32)
    key = Nx.Random.key(0)

    assert_raise ArgumentError,
                 "expected the number of steps to be less or equal to the number of training steps (1000), got: 1001",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 1001, sample_template, key)
                 end

    assert_raise ArgumentError,
                 "expected the number of steps to be less or equal to num_original_steps * strength (50 * 0.5). Either reduce the number of steps orincrease the strength",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 50, sample_template, key,
                     strength: 0.5
                   )
                 end

    assert_raise ArgumentError,
                 "expected the number of steps to be less or equal to the number of original steps (50), got: 51",
                 fn ->
                   Bumblebee.Diffusion.LcmScheduler.init(scheduler, 51, sample_template, key)
                 end
  end

  test "timesteps" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler)

    timesteps = scheduler_timesteps(scheduler, 10)
    assert_equal(timesteps, Nx.tensor([999, 899, 799, 699, 599, 499, 399, 299, 199, 99]))

    timesteps = scheduler_timesteps(scheduler, 3)
    assert_equal(timesteps, Nx.tensor([999, 679, 339]))

    timesteps = scheduler_timesteps(scheduler, 1)
    assert_equal(timesteps, Nx.tensor([999]))
  end

  # Note that we only test a single step, because more steps depend on
  # randomization and we have different PRNG implementation than the
  # reference implementation

  test "default configuration" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler)

    sample = scheduler_loop(scheduler, 1)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0071, 0.0072], [0.0078, 0.0080], [0.0086, 0.0088]],
        [[0.0131, 0.0133], [0.0139, 0.0141], [0.0147, 0.0149]],
        [[0.0192, 0.0194], [0.0200, 0.0202], [0.0208, 0.0210]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(6.2203))
  end

  test ":linear beta schedule" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler, beta_schedule: :linear)

    sample = scheduler_loop(scheduler, 1)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0065, 0.0067], [0.0072, 0.0074], [0.0079, 0.0081]],
        [[0.0121, 0.0123], [0.0128, 0.0130], [0.0135, 0.0137]],
        [[0.0178, 0.0179], [0.0185, 0.0186], [0.0192, 0.0193]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(5.7404))
  end

  test ":squared_cosine beta schedule" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler, beta_schedule: :squared_cosine)

    sample = scheduler_loop(scheduler, 1)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[2.9326, 3.0118], [3.2498, 3.3290], [3.5667, 3.6462]],
        [[5.4692, 5.5484], [5.7860, 5.8653], [6.1035, 6.1827]],
        [[8.0054, 8.0846], [8.3229, 8.4021], [8.6397, 8.7190]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(2587.1404))
  end

  test "beta schedule range" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler, beta_start: 0.001, beta_end: 0.02)

    sample = scheduler_loop(scheduler, 1)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0114, 0.0117], [0.0126, 0.0129], [0.0138, 0.0141]],
        [[0.0212, 0.0215], [0.0224, 0.0227], [0.0237, 0.0240]],
        [[0.0310, 0.0313], [0.0323, 0.0326], [0.0335, 0.0338]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(10.0290))
  end

  test ":angular_velocity prediction type" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.LcmScheduler, prediction_type: :angular_velocity)

    sample = scheduler_loop(scheduler, 1)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[-0.1342, -0.1378], [-0.1487, -0.1523], [-0.1632, -0.1668]],
        [[-0.2502, -0.2539], [-0.2647, -0.2684], [-0.2792, -0.2829]],
        [[-0.3663, -0.3699], [-0.3808, -0.3844], [-0.3953, -0.3989]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(-118.3716))
  end
end
