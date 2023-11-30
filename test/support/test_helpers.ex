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

  def to_channels_first(tensor) do
    Nx.transpose(tensor, axes: [0, 3, 1, 2])
  end
end
