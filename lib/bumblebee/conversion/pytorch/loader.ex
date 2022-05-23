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
    {:ok, contents} = :zip.unzip(String.to_charlist(path), [:memory])

    contents =
      Map.new(contents, fn {name, content} ->
        {List.to_string(name), content}
      end)

    {term, ""} =
      Unpickler.load!(contents["archive/data.pkl"],
        object_resolver: &object_resolver/1,
        persistent_id_resolver: fn
          {"storage", storage_type, key, _location, _size} ->
            binary = Map.fetch!(contents, "archive/data/#{key}")
            {:storage, storage_type, binary}
        end
      )

    term
  end

  defp object_resolver(%{constructor: "collections.OrderedDict", set_items: items}) do
    {:ok, Map.new(items)}
  end

  # See https://github.com/pytorch/pytorch/blob/v1.11.0/torch/_tensor.py#L271-L280
  defp object_resolver(%{
         constructor: "torch._utils._rebuild_tensor_v2",
         args: [storage, offset, shape, strides, _requires_grad, _backward_hooks]
       }) do
    {:storage, storage_type, binary} = storage
    {_, bit_unit} = type = to_nx_type(storage_type)
    byte_unit = div(bit_unit, 8)
    size = Tuple.product(shape)
    binary = binary_part(binary, offset * byte_unit, size * byte_unit)
    tensor = binary |> Nx.from_binary(type) |> to_contiguous(shape, strides)
    {:ok, tensor}
  end

  defp object_resolver(_object), do: :error

  defp to_nx_type(%Unpickler.Global{scope: "torch", name: name}) do
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

  defp to_contiguous(tensor, shape, strides) do
    # PyTorch tensors may not be contiguous in memory, so strides are
    # used to indicate jumps necessary when traversing each axis.
    # Since Nx doesn't have the notion of strides, we transpose the
    # tensor, in a way that makes it contiguous, which is equivalent
    # to strides being decreasing

    axes_order =
      strides
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.sort_by(&elem(&1, 0), :desc)
      |> Enum.map(&elem(&1, 1))

    if axes_order == Nx.axes(shape) do
      Nx.reshape(tensor, shape)
    else
      memory_shape =
        axes_order
        |> Enum.with_index()
        |> Enum.reduce(shape, fn {memory_shape_idx, shape_idx}, memory_shape ->
          put_elem(memory_shape, memory_shape_idx, elem(shape, shape_idx))
        end)

      tensor |> Nx.reshape(memory_shape) |> Nx.transpose(axes: axes_order)
    end
  end

  @legacy_magic_number 119_547_037_146_038_801_333_356

  defp load_legacy!(path) do
    data = File.read!(path)

    {@legacy_magic_number, data} = Unpickler.load!(data)
    {_protocol_version, data} = Unpickler.load!(data)
    {_system_info, data} = Unpickler.load!(data)

    binaries = storage_binaries(data)

    {term, _} =
      Unpickler.load!(data,
        object_resolver: &object_resolver/1,
        persistent_id_resolver: fn
          {"storage", storage_type, root_key, _location, _size, view_metadata} ->
            {_, bit_unit} = to_nx_type(storage_type)
            byte_unit = div(bit_unit, 8)

            binary =
              case view_metadata do
                nil ->
                  binaries[root_key]

                {_view_key, offset, view_size} ->
                  binary_part(binaries[root_key], offset * byte_unit, view_size * byte_unit)
              end

            {:storage, storage_type, binary}

          {"module", module, _source_file, _source} ->
            module
        end
      )

    term
  end

  defp storage_binaries(data) do
    # We first do a dry run on the pickle and extract storage metadata,
    # then we use that metadata to read the storage binaries that follow

    {term, data} =
      Unpickler.load!(data,
        persistent_id_resolver: fn
          {"storage", storage_type, root_key, _location, size, _view_metadata} ->
            {_, bit_unit} = to_nx_type(storage_type)
            byte_unit = div(bit_unit, 8)
            {:storage_info, root_key, {size, byte_unit}}

          _other ->
            nil
        end
      )

    storage_infos = collect_storage_infos(term, %{})

    {storage_keys, data} = Unpickler.load!(data)

    {pairs, ""} =
      Enum.map_reduce(storage_keys, data, fn key, data ->
        {size, byte_unit} = Map.fetch!(storage_infos, key)
        bytes = size * byte_unit

        # Each storage binary is prefixed with the number of elements.
        # See https://github.com/pytorch/pytorch/blob/v1.11.0/torch/csrc/generic/serialization.cpp#L93-L134
        <<^size::integer-little-signed-size(64), chunk::binary-size(bytes), data::binary>> = data

        {{key, chunk}, data}
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
