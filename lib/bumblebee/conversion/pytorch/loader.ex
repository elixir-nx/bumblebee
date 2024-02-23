defmodule Bumblebee.Conversion.PyTorch.Loader do
  @moduledoc false

  @doc """
  Loads data saved using [`torch.save`](https://pytorch.org/docs/stable/generated/torch.save.html).

  PyTorch tensors are automatically deserialized as Nx tensors.

  This function supports both the current zip-based file format and
  the legacy format.
  """
  @spec load!(Path.t()) :: term()
  def load!(path) do
    # See https://github.com/pytorch/pytorch/blob/v1.11.0/torch/serialization.py#L607

    if zip?(path) do
      load_zip!(path)
    else
      load_legacy!(path)
    end
  end

  defp zip?(path) do
    # Check for the "local file header signature"
    File.open(path, [:read], &IO.binread(&1, 4)) == {:ok, <<80, 75, 3, 4>>}
  end

  defp load_zip!(path) do
    open_zip!(path, fn unzip ->
      file_name_map =
        unzip
        |> Unzip.list_entries()
        |> Map.new(fn %Unzip.Entry{file_name: file_name} ->
          # Strip the root dir from the file name
          name = file_name |> Path.split() |> Enum.drop(1) |> Enum.join("/")
          {name, file_name}
        end)

      binary = read_zip_file(unzip, Map.fetch!(file_name_map, "data.pkl"))

      {term, ""} =
        Unpickler.load!(binary,
          object_resolver: &object_resolver/1,
          persistent_id_resolver: fn
            {"storage", storage_type, key, _location, _size} ->
              file_name = Map.fetch!(file_name_map, "data/#{key}")
              {:storage, storage_type, {:zip, path, file_name}}
          end
        )

      term
    end)
  end

  @doc false
  def open_zip!(path, fun) do
    zip_file = Unzip.LocalFile.open(path)

    try do
      {:ok, unzip} = Unzip.new(zip_file)
      fun.(unzip)
    after
      Unzip.LocalFile.close(zip_file)
    end
  end

  @doc false
  def read_zip_file(unzip, file_name) do
    unzip
    |> Unzip.file_stream!(file_name)
    |> Enum.to_list()
    |> IO.iodata_to_binary()
  end

  defp object_resolver(%{constructor: "collections.OrderedDict", set_items: items}) do
    {:ok, Map.new(items)}
  end

  # See https://github.com/pytorch/pytorch/blob/v1.11.0/torch/_tensor.py#L271-L280
  defp object_resolver(%{
         constructor: "torch._utils._rebuild_tensor_v2",
         args: [storage, offset, shape, strides, _requires_grad, _backward_hooks]
       }) do
    {:storage, storage_type, storage} = storage
    type = storage_type_to_nx(storage_type)

    lazy_tensor = %Bumblebee.Conversion.PyTorch.FileTensor{
      shape: shape,
      type: type,
      offset: offset,
      strides: strides,
      storage: storage
    }

    {:ok, lazy_tensor}
  end

  # See https://github.com/pytorch/pytorch/blob/v1.12.1/torch/_tensor.py#L222-L226
  defp object_resolver(%{
         constructor: "torch._utils._rebuild_device_tensor_from_numpy",
         args: [numpy_array, _type, _device, _requires_grad]
       }) do
    # Tensors on certain devices are serialized as numpy arrays
    {:ok, numpy_array}
  end

  # See https://github.com/numpy/numpy/blob/v1.23.3/numpy/core/src/multiarray/descriptor.c#L2506-L2508
  defp object_resolver(%{constructor: "numpy.dtype", args: [type, false = _align, true = _copy]}) do
    {:ok, numpy_type_to_nx(type)}
  end

  # See https://github.com/numpy/numpy/blob/v1.23.3/numpy/core/src/multiarray/methods.c#L2009-L2014
  defp object_resolver(%{
         constructor: "numpy.core.multiarray._reconstruct",
         state: {_version, shape, type, fortran_order?, data}
       }) do
    tensor =
      if fortran_order? do
        # In Fortran order the axes are reversed in memory
        shape =
          shape
          |> Tuple.to_list()
          |> Enum.reverse()
          |> List.to_tuple()

        data |> Nx.from_binary(type) |> Nx.reshape(shape) |> Nx.transpose()
      else
        data |> Nx.from_binary(type) |> Nx.reshape(shape)
      end

    {:ok, tensor}
  end

  defp object_resolver(_object), do: :error

  defp storage_type_to_nx(%Unpickler.Global{scope: "torch", name: name}) do
    # See https://github.com/pytorch/pytorch/blob/v1.11.0/torch/storage.py#L189-L208
    mapping = %{
      "DoubleStorage" => {:f, 64},
      "FloatStorage" => {:f, 32},
      "HalfStorage" => {:f, 16},
      "LongStorage" => {:s, 64},
      "IntStorage" => {:s, 32},
      "ShortStorage" => {:s, 16},
      "CharStorage" => {:s, 8},
      "ByteStorage" => {:u, 8},
      "BoolStorage" => {:u, 8},
      "BFloat16Storage" => {:bf, 16},
      "ComplexDoubleStorage" => {:c, 128},
      "ComplexFloatStorage" => {:c, 64}
    }

    if type = mapping[name] do
      type
    else
      raise "unsupported PyTorch storage type: #{inspect(name)}"
    end
  end

  defp numpy_type_to_nx(name) do
    # See https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html#numpy-dtype-kind
    mapping = %{
      "b1" => {:u, 8},
      "i1" => {:s, 8},
      "i2" => {:s, 16},
      "i4" => {:s, 32},
      "i8" => {:s, 64},
      "u1" => {:u, 8},
      "u2" => {:u, 16},
      "u4" => {:u, 32},
      "u8" => {:u, 64},
      "f2" => {:f, 16},
      "f4" => {:f, 32},
      "f8" => {:f, 64},
      "c8" => {:c, 64},
      "c16" => {:c, 128}
    }

    if type = mapping[name] do
      type
    else
      raise "unsupported NumPy data type: #{inspect(name)}"
    end
  end

  @legacy_magic_number 119_547_037_146_038_801_333_356

  defp load_legacy!(path) do
    data = File.read!(path)
    full_size = byte_size(data)

    {@legacy_magic_number, data} = Unpickler.load!(data)
    {_protocol_version, data} = Unpickler.load!(data)
    {_system_info, data} = Unpickler.load!(data)

    binaries = storage_binaries(data, full_size)

    {term, _} =
      Unpickler.load!(data,
        object_resolver: &object_resolver/1,
        persistent_id_resolver: fn
          {"storage", storage_type, root_key, _location, _size, view_metadata} ->
            {_, bit_unit} = storage_type_to_nx(storage_type)
            byte_unit = div(bit_unit, 8)

            {file_offset, size} = Map.fetch!(binaries, root_key)

            storage =
              case view_metadata do
                nil ->
                  {:file, path, file_offset, size}

                {_view_key, offset, view_size} ->
                  {:file, path, file_offset + offset * byte_unit, view_size * byte_unit}
              end

            {:storage, storage_type, storage}

          {"module", module, _source_file, _source} ->
            module
        end
      )

    term
  end

  defp storage_binaries(data, full_size) do
    # We first do a dry run on the pickle and extract storage metadata,
    # then we use that metadata to read the storage binaries that follow

    {term, data} =
      Unpickler.load!(data,
        persistent_id_resolver: fn
          {"storage", storage_type, root_key, _location, size, _view_metadata} ->
            {_, bit_unit} = storage_type_to_nx(storage_type)
            byte_unit = div(bit_unit, 8)
            {:storage_info, root_key, {size, byte_unit}}

          _other ->
            nil
        end
      )

    storage_infos = collect_storage_infos(term, %{})

    {storage_keys, data} = Unpickler.load!(data)

    offset = full_size - byte_size(data)

    {pairs, _offset} =
      Enum.map_reduce(storage_keys, offset, fn key, offset ->
        {size, byte_unit} = Map.fetch!(storage_infos, key)
        bytes = size * byte_unit

        # Each storage binary is prefixed with the number of elements,
        # stored as integer-little-signed-size(64), hence the 8 bytes.
        # See https://github.com/pytorch/pytorch/blob/v1.11.0/torch/csrc/generic/serialization.cpp#L93-L134
        start_offset = offset + 8
        offset = start_offset + bytes

        {{key, {start_offset, bytes}}, offset}
      end)

    Map.new(pairs)
  end

  defp collect_storage_infos({:storage_info, key, meta}, storage_infos) do
    Map.put(storage_infos, key, meta)
  end

  defp collect_storage_infos(list, storages) when is_list(list) do
    Enum.reduce(list, storages, &collect_storage_infos/2)
  end

  defp collect_storage_infos(%MapSet{} = set, storages) do
    Enum.reduce(set, storages, &collect_storage_infos/2)
  end

  defp collect_storage_infos(%_{} = struct, storages) do
    struct |> Map.values() |> collect_storage_infos(storages)
  end

  defp collect_storage_infos(%{} = map, storages) do
    Enum.reduce(map, storages, &collect_storage_infos/2)
  end

  defp collect_storage_infos(tuple, storages) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> collect_storage_infos(storages)
  end

  defp collect_storage_infos(_term, storages) do
    storages
  end
end
