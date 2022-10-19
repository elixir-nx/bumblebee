defmodule Bumblebee.Utils do
  @moduledoc false

  @doc """
  Formats the given number of bytes into a human-friendly text.

  ## Examples

      iex> Bumblebee.Utils.format_bytes(0)
      "0 B"

      iex> Bumblebee.Utils.format_bytes(900)
      "900 B"

      iex> Bumblebee.Utils.format_bytes(1100)
      "1.1 KB"

      iex> Bumblebee.Utils.format_bytes(1_228_800)
      "1.2 MB"

      iex> Bumblebee.Utils.format_bytes(1_363_148_800)
      "1.4 GB"

      iex> Bumblebee.Utils.format_bytes(1_503_238_553_600)
      "1.5 TB"

  """
  @spec format_bytes(non_neg_integer()) :: String.t()
  def format_bytes(bytes) when is_integer(bytes) do
    cond do
      bytes >= memory_unit(:TB) -> format_bytes(bytes, :TB)
      bytes >= memory_unit(:GB) -> format_bytes(bytes, :GB)
      bytes >= memory_unit(:MB) -> format_bytes(bytes, :MB)
      bytes >= memory_unit(:KB) -> format_bytes(bytes, :KB)
      true -> format_bytes(bytes, :B)
    end
  end

  defp format_bytes(bytes, :B) when is_integer(bytes), do: "#{bytes} B"

  defp format_bytes(bytes, unit) when is_integer(bytes) do
    value = bytes / memory_unit(unit)
    "#{:erlang.float_to_binary(value, decimals: 1)} #{unit}"
  end

  defp memory_unit(:TB), do: 1_000_000_000_000
  defp memory_unit(:GB), do: 1_000_000_000
  defp memory_unit(:MB), do: 1_000_000
  defp memory_unit(:KB), do: 1_000
end
