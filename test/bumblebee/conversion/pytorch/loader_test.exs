defmodule Bumblebee.Conversion.PyTorch.LoaderTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Conversion.PyTorch.Loader

  setup do
    Nx.default_backend(Nx.BinaryBackend)
    :ok
  end

  @dir Path.expand("../../../fixtures/pytorch", __DIR__)

  for format <- ["zip", "legacy"] do
    @format format

    describe "#{format} format" do
      test "tensors" do
        path = Path.join(@dir, "tensors.#{@format}.pt")

        assert Loader.load!(path) == [
                 Nx.tensor([-1.0, 1.0], type: :f64),
                 Nx.tensor([-1.0, 1.0], type: :f32),
                 Nx.tensor([-1.0, 1.0], type: :f16),
                 Nx.tensor([-1, 1], type: :s64),
                 Nx.tensor([-1, 1], type: :s32),
                 Nx.tensor([-1, 1], type: :s16),
                 Nx.tensor([-1, 1], type: :s8),
                 Nx.tensor([0, 1], type: :u8),
                 Nx.tensor([-1.0, 1.0], type: :bf16),
                 Nx.tensor([Complex.new(1, -1), Complex.new(1, 1)], type: :c128),
                 Nx.tensor([Complex.new(1, -1), Complex.new(1, 1)], type: :c64)
               ]
      end

      test "numpy arrays" do
        path = Path.join(@dir, "numpy_arrays.#{@format}.pt")

        assert Loader.load!(path) == [
                 Nx.tensor([-1.0, 1.0], type: :f64),
                 Nx.tensor([-1.0, 1.0], type: :f32),
                 Nx.tensor([-1.0, 1.0], type: :f16),
                 Nx.tensor([-1, 1], type: :s64),
                 Nx.tensor([-1, 1], type: :s32),
                 Nx.tensor([-1, 1], type: :s16),
                 Nx.tensor([-1, 1], type: :s8),
                 Nx.tensor([0, 1], type: :u64),
                 Nx.tensor([0, 1], type: :u32),
                 Nx.tensor([0, 1], type: :u16),
                 Nx.tensor([0, 1], type: :u8),
                 Nx.tensor([0, 1], type: :u8),
                 Nx.tensor([Complex.new(1, -1), Complex.new(1, 1)], type: :c128),
                 Nx.tensor([Complex.new(1, -1), Complex.new(1, 1)], type: :c64)
               ]
      end

      test "ordered dict" do
        path = Path.join(@dir, "ordered_dict.#{@format}.pt")

        assert Loader.load!(path) == %{"x" => 1, "y" => 2}
      end

      test "noncontiguous tensor" do
        path = Path.join(@dir, "noncontiguous_tensor.#{@format}.pt")

        assert Loader.load!(path) ==
                 Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], type: :s64)
      end

      test "numpy array in Fortran order" do
        path = Path.join(@dir, "noncontiguous_numpy_array.#{@format}.pt")

        assert Loader.load!(path) ==
                 Nx.tensor([[1, 4], [2, 5], [3, 6]], type: :s64)
      end
    end
  end

  test "legacy format storage view" do
    # Note that storage views have been removed in PyTorch v0.4.0,
    # this test is based on https://github.com/pytorch/pytorch/blob/v1.11.0/test/test_serialization.py#L554-L575
    path = Path.join(@dir, "storage_view.legacy.pt")

    assert Loader.load!(path) ==
             {{:storage, %Unpickler.Global{scope: "torch", name: "FloatStorage"}, <<0, 0, 0, 0>>},
              {:storage, %Unpickler.Global{scope: "torch", name: "FloatStorage"}, <<0, 0, 0, 0>>}}
  end

  test "raises if the files does not exist" do
    assert_raise File.Error, ~r/no such file or directory/, fn ->
      Loader.load!("nonexistent")
    end
  end
end
