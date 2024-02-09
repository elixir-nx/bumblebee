defmodule Bumblebee.Shared.Converters do
  @moduledoc false

  @type converter ::
          (name :: String.t(), value :: term() ->
             {:ok, term()} | {:error, String.t()})

  @doc """
  Reads and converts data from the given map according to the given
  specification.
  """
  @spec convert!(map(), keyword({String.t(), converter()})) :: keyword()
  def convert!(data, entries) do
    for {key, {source, converter}} <- entries, Map.has_key?(data, source) do
      case converter.(source, data[source]) do
        {:ok, value} ->
          {key, value}

        {:error, message} ->
          raise "conversion failed, " <> message
      end
    end
  end

  def number() do
    fn name, value ->
      if is_number(value) do
        {:ok, value}
      else
        {:error, "expected #{inspect(name)} to be a number, got: #{inspect(value)}"}
      end
    end
  end

  def string() do
    fn name, value ->
      if is_binary(value) do
        {:ok, value}
      else
        {:error, "expected #{inspect(name)} to be a string, got: #{inspect(value)}"}
      end
    end
  end

  def atom() do
    fn name, value ->
      try do
        {:ok, String.to_atom(value)}
      rescue
        _error ->
          {:error, "unsupported value for #{inspect(name)}, got: #{inspect(value)}"}
      end
    end
  end

  def boolean() do
    fn name, value ->
      if is_boolean(value) do
        {:ok, value}
      else
        {:error, "expected #{inspect(name)} to be a boolean, got: #{inspect(value)}"}
      end
    end
  end

  def integer_as_string() do
    fn name, value ->
      with true <- is_binary(value), {number, ""} <- Integer.parse(value) do
        {:ok, number}
      else
        _ ->
          {:error,
           "expected #{inspect(name)} to be a string representing a number, got: #{inspect(value)}"}
      end
    end
  end

  def list(converter) do
    fn name, value ->
      if is_list(value) do
        value
        |> Enum.with_index()
        |> safe_map(fn {item, index} ->
          converter.("#{name}[#{index}]", item)
        end)
      else
        {:error, "expected #{inspect(name)} to be a list, got: #{inspect(value)}"}
      end
    end
  end

  def map(key_converter, value_converter) do
    fn name, value ->
      if is_map(value) do
        value
        |> safe_map(fn {key, value} ->
          with {:ok, key} <- key_converter.("#{name} key", key),
               {:ok, value} <- value_converter.("#{name}.#{value}]", value),
               do: {:ok, {key, value}}
        end)
        |> case do
          {:ok, list} -> {:ok, Map.new(list)}
          error -> error
        end
      else
        {:error, "expected #{inspect(name)} to be a map, got: #{inspect(value)}"}
      end
    end
  end

  def tuple(converters) do
    fn name, value ->
      cond do
        not is_list(value) ->
          {:error, "expected #{inspect(name)} to be a list, got: #{inspect(value)}"}

        length(value) != length(converters) ->
          {:error,
           "expected #{inspect(name)} to have #{inspect(length(converters))} elements," <>
             " but got #{inspect(length(value))}"}

        true ->
          value
          |> Enum.with_index()
          |> Enum.zip(converters)
          |> safe_map(fn {{item, index}, converter} ->
            converter.("#{name}[#{index}]", item)
          end)
          |> case do
            {:ok, list} -> {:ok, List.to_tuple(list)}
            error -> error
          end
      end
    end
  end

  defp safe_map(enumerable, fun) do
    enumerable
    |> Enum.reduce_while([], fn value, acc ->
      case fun.(value) do
        {:ok, value} -> {:cont, [value | acc]}
        {:error, error} -> {:halt, {:error, error}}
      end
    end)
    |> case do
      {:error, error} -> {:error, error}
      items -> {:ok, Enum.reverse(items)}
    end
  end

  def mapping(map) do
    fn name, value ->
      case Map.fetch(map, value) do
        {:ok, replacement} ->
          {:ok, replacement}

        :error ->
          {:error,
           "unrecognized value for #{inspect(name)}, got: #{inspect(value)}," <>
             " but the supported values are: #{inspect(Map.keys(map))}"}
      end
    end
  end

  def optional(converter) do
    fn name, value ->
      if value == nil do
        {:ok, nil}
      else
        converter.(name, value)
      end
    end
  end

  def one_of(converters) do
    fn name, value ->
      default =
        {:error,
         "value #{inspect(value)} at #{inspect(name)} did not match any of the expected formats"}

      Enum.find_value(converters, default, fn converter ->
        with {:error, _} <- converter.(name, value), do: nil
      end)
    end
  end

  def resize_method() do
    # See https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling
    mapping(%{
      0 => :nearest,
      1 => :lanczos3,
      2 => :bilinear,
      3 => :bicubic
    })
  end

  def padding(rank) do
    fn name, value ->
      if is_number(value) do
        {:ok, List.duplicate({value, value}, rank)}
      else
        {:error, "expected #{inspect(name)} to be a number, got: #{inspect(value)}"}
      end
    end
  end

  def activation() do
    mapping = %{
      "gelu_new" => :gelu_approx_tanh,
      "gelu_pytorch_tanh" => :gelu_approx_tanh,
      "quick_gelu" => :gelu_approx_sigmoid
    }

    fn name, value ->
      case Map.fetch(mapping, value) do
        {:ok, replacement} -> {:ok, replacement}
        :error -> atom().(name, value)
      end
    end
  end

  def image_size(opts \\ []) do
    opts = Keyword.validate!(opts, single_as: :both_edges)
    single_as = opts[:single_as]

    true = single_as in [:both_edges, :shortest_edge]

    fn name, value ->
      case value do
        %{"height" => height, "width" => width} ->
          {:ok, %{height: height, width: width}}

        [height, width] ->
          {:ok, %{height: height, width: width}}

        size when is_number(size) and single_as == :both_edges ->
          {:ok, %{height: size, width: size}}

        size when is_number(size) and single_as == :shortest_edge ->
          {:ok, %{shortest_edge: size}}

        %{"shortest_edge" => shortest_edge} ->
          {:ok, %{shortest_edge: shortest_edge}}

        _ ->
          {:error,
           "expected #{inspect(name)} to be a number, a list or a map with height and width, got: #{inspect(value)}"}
      end
    end
  end
end
