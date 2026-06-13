defmodule Bumblebee.Utils.ChunkedFileTensor do
  @moduledoc false

  # A lazy tensor that reads from a safetensors shard file in chunks.
  # Replaces Safetensors.FileTensor for large tensors to work around macOS's
  # pread(2) EINVAL when the byte count exceeds INT_MAX (~2 GB).
  defstruct [:path, :byte_offset, :byte_size, :shape, :type]

  # 1 GB chunks — safely below INT_MAX on macOS.
  @max_chunk_size 1_073_741_824

  @doc """
  Wraps a `Safetensors.FileTensor` in a `ChunkedFileTensor`.
  """
  def from_file_tensor(%Safetensors.FileTensor{} = ft) do
    %__MODULE__{
      path: ft.path,
      byte_offset: ft.byte_offset,
      byte_size: ft.byte_size,
      shape: ft.shape,
      type: ft.type
    }
  end

  @doc """
  Reads `size` bytes at `offset` from an already-open raw file handle,
  splitting into ≤1 GB reads to avoid the macOS pread EINVAL limit.
  """
  def read_chunked(file, offset, size) when size <= @max_chunk_size do
    {:ok, binary} = :file.pread(file, offset, size)
    binary
  end

  def read_chunked(file, offset, size) do
    full_chunks = div(size, @max_chunk_size)
    remainder = rem(size, @max_chunk_size)

    chunks =
      for i <- 0..(full_chunks - 1) do
        {:ok, chunk} = :file.pread(file, offset + i * @max_chunk_size, @max_chunk_size)
        chunk
      end

    chunks =
      if remainder > 0 do
        {:ok, tail} = :file.pread(file, offset + full_chunks * @max_chunk_size, remainder)
        chunks ++ [tail]
      else
        chunks
      end

    IO.iodata_to_binary(chunks)
  end
end

defimpl Nx.LazyContainer, for: Bumblebee.Utils.ChunkedFileTensor do
  alias Bumblebee.Utils.ChunkedFileTensor

  def traverse(lazy, acc, fun) do
    template = Nx.template(lazy.shape, lazy.type)

    load = fn ->
      File.open!(lazy.path, [:read, :raw], fn file ->
        binary = ChunkedFileTensor.read_chunked(file, lazy.byte_offset, lazy.byte_size)
        Safetensors.Shared.build_tensor(binary, lazy.shape, lazy.type)
      end)
    end

    fun.(template, load, acc)
  end
end
