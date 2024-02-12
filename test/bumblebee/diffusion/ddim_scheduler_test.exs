defmodule Bumblebee.Diffusion.DdimSchedulerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test "timesteps" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.DdimScheduler)

    timesteps = scheduler_timesteps(scheduler, 10)

    assert_equal(timesteps, Nx.tensor([900, 800, 700, 600, 500, 400, 300, 200, 100, 0]))
  end

  test "default configuration" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.DdimScheduler)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0648, 0.0666], [0.0718, 0.0736], [0.0788, 0.0806]],
        [[0.1209, 0.1226], [0.1279, 0.1296], [0.1349, 0.1367]],
        [[0.1770, 0.1787], [0.1840, 0.1857], [0.1910, 0.1927]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(57.1861))
  end

  test ":quadratic beta schedule" do
    scheduler = Bumblebee.configure(Bumblebee.Diffusion.DdimScheduler, beta_schedule: :quadratic)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0675, 0.0693], [0.0747, 0.0766], [0.0820, 0.0839]],
        [[0.1258, 0.1276], [0.1331, 0.1349], [0.1404, 0.1422]],
        [[0.1841, 0.1860], [0.1914, 0.1932], [0.1987, 0.2005]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(59.5049))
  end

  test ":squared_cosine beta schedule" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.DdimScheduler, beta_schedule: :squared_cosine)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0684, 0.0702], [0.0757, 0.0776], [0.0831, 0.0850]],
        [[0.1275, 0.1293], [0.1349, 0.1367], [0.1422, 0.1441]],
        [[0.1866, 0.1884], [0.1940, 0.1958], [0.2014, 0.2032]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(60.2983))
  end

  test "beta schedule range" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.DdimScheduler, beta_start: 0.001, beta_end: 0.02)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0653, 0.0670], [0.0723, 0.0741], [0.0794, 0.0812]],
        [[0.1217, 0.1235], [0.1288, 0.1306], [0.1359, 0.1376]],
        [[0.1782, 0.1800], [0.1853, 0.1870], [0.1923, 0.1941]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(57.5869))
  end

  test ":angular_velocity prediction type" do
    scheduler =
      Bumblebee.configure(Bumblebee.Diffusion.DdimScheduler, prediction_type: :angular_velocity)

    sample = scheduler_loop(scheduler, 10)

    assert_all_close(
      sample[[1..3, 1..3, 1..2]],
      Nx.tensor([
        [[0.0198, 0.0203], [0.0219, 0.0225], [0.0241, 0.0246]],
        [[0.0369, 0.0375], [0.0391, 0.0396], [0.0412, 0.0417]],
        [[0.0540, 0.0546], [0.0562, 0.0567], [0.0583, 0.0589]]
      ])
    )

    assert_all_close(Nx.sum(sample), Nx.tensor(17.4644))
  end
end
