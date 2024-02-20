defmodule Bumblebee.TestHelpers do
  @moduledoc false

  import ExUnit.Assertions

  defmacro assert_equal(left, right) do
    # Assert against binary backend tensors to show diff on failure
    quote do
      left = unquote(left) |> to_binary_backend()
      right = unquote(right) |> Nx.as_type(Nx.type(left)) |> to_binary_backend()
      assert left == right
    end
  end

  def to_binary_backend(tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  def assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    equals =
      left
      |> Nx.all_close(right, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if equals != Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend) do
      flunk("""
      expected

      #{inspect(left)}

      to be within tolerance of

      #{inspect(right)}
      """)
    end
  end

  def model_test_tags() do
    [model: true, capture_log: true, timeout: 60_000]
  end

  def serving_test_tags() do
    [serving: true, slow: true, capture_log: true, timeout: 600_000]
  end

  def scheduler_loop(scheduler, num_steps) do
    sample = dummy_sample()

    {state, timesteps} =
      Bumblebee.scheduler_init(scheduler, num_steps, Nx.to_template(sample), Nx.Random.key(0))

    {_state, sample} =
      for i <- 0..(Nx.size(timesteps) - 1), reduce: {state, sample} do
        {state, sample} ->
          prediction = dummy_model(sample, timesteps[i])
          Bumblebee.scheduler_step(scheduler, state, sample, prediction)
      end

    sample
  end

  def scheduler_timesteps(scheduler, num_steps) do
    sample = dummy_sample()

    {_state, timesteps} =
      Bumblebee.scheduler_init(scheduler, num_steps, Nx.to_template(sample), Nx.Random.key(0))

    timesteps
  end

  defp dummy_sample() do
    shape = {_height = 8, _width = 8, _channels = 4}
    sample = Nx.iota(shape)
    Nx.divide(sample, Nx.size(sample))
  end

  defp dummy_model(sample, timestep) do
    sample
    |> Nx.multiply(timestep)
    |> Nx.divide(Nx.add(timestep, 1))
  end
end
