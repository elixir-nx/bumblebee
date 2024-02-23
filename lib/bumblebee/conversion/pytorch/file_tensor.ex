defmodule Bumblebee.Conversion.PyTorch.FileTensor do
  @moduledoc false

  defstruct [:shape, :type, :offset, :strides, :storage]
end

defimpl Nx.LazyContainer, for: Bumblebee.Conversion.PyTorch.FileTensor do
  alias Bumblebee.Conversion.PyTorch.Loader

  def traverse(lazy_tensor, acc, fun) do
    template = Nx.template(lazy_tensor.shape, lazy_tensor.type)

    load = fn ->
      binary =
        case lazy_tensor.storage do
          {:zip, path, file_name} ->
            Loader.open_zip!(path, fn unzip ->
              Loader.read_zip_file(unzip, file_name)
            end)

          {:file, path, offset, size} ->
            File.open!(path, [:read, :raw], fn file ->
              {:ok, binary} = :file.pread(file, offset, size)
              binary
            end)
        end

      %{offset: offset, shape: shape, type: type, strides: strides} = lazy_tensor

      {_, bit_unit} = type
      byte_unit = div(bit_unit, 8)
      size = Tuple.product(shape)
      binary = binary_part(binary, offset * byte_unit, size * byte_unit)
      binary |> Nx.from_binary(type) |> to_contiguous(shape, strides)
    end

    fun.(template, load, acc)
  end

  defp to_contiguous(tensor, shape, strides) do
    # PyTorch tensors may not be contiguous in memory, so strides are
    # used to indicate jumps necessary when traversing each axis.
    # Since Nx doesn't have the notion of strides, we transpose the
    # tensor, in a way that makes it contiguous, which is equivalent
    # to strides being decreasing

    memory_axes_order =
      strides
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.sort_by(&elem(&1, 0), :desc)
      |> Enum.map(&elem(&1, 1))

    if memory_axes_order == Nx.axes(shape) do
      Nx.reshape(tensor, shape)
    else
      memory_shape =
        memory_axes_order
        |> Enum.map(fn axis -> elem(shape, axis) end)
        |> List.to_tuple()

      tensor
      |> Nx.reshape(memory_shape)
      |> Nx.transpose(axes: inverse_permutation(memory_axes_order))
    end
  end

  defp inverse_permutation(list) do
    list
    |> Enum.with_index()
    |> Enum.reduce(List.to_tuple(list), fn {src_idx, dest_idx}, inverse ->
      put_elem(inverse, src_idx, dest_idx)
    end)
    |> Tuple.to_list()
  end
end
