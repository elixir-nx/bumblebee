defmodule Bumblebee.Utils.ProgressBar do
  @moduledoc false

  # Reserve 2 chars for the start and end of the bar
  # and one to leave a gap at the end of a line, to prevent wrapping on to the next line
  @reserved_width 3

  @start_end_char "|"
  @filled_char "="
  @unfilled_char " "
  @carriage_return "\r"

  @default_opts [unit: :none]

  @doc """
  Renders a simple progress bar to the terminal.
  The progress bar sizes to fill the entire width of the terminal.
  """
  @spec render(number(), number(), keyword()) :: :ok
  def render(count, total, opts \\ @default_opts) do
    opts = Keyword.merge(@default_opts, opts)
    unit = Keyword.get(opts, :unit, :none)
    percent = min(max(count / total, 0), 1)

    formatted_percent = String.pad_leading("#{trunc(percent * 100)}%", 4)
    formatted_progress = " " <> formatted_progress(count, total, unit)

    reserved_width =
      @reserved_width + byte_size(formatted_progress) + byte_size(formatted_percent)

    bar_width = max(terminal_width() - reserved_width, 0)

    filled_char_count = trunc(percent * bar_width)
    unfilled_char_count = max(bar_width - filled_char_count, 0)

    filled_chars = String.duplicate(@filled_char, filled_char_count)
    unfilled_chars = String.duplicate(@unfilled_char, unfilled_char_count)

    output =
      @carriage_return <>
        @start_end_char <>
        filled_chars <>
        unfilled_chars <>
        @start_end_char <>
        formatted_percent <>
        formatted_progress

    IO.write(output)
  end

  defp formatted_progress(progress, total, :none) do
    "#{progress}/#{total}"
  end

  defp formatted_progress(progress, total, :bytes) do
    {unit, divisor} = bytes_unit_and_divisor(total)
    "#{trunc(progress / divisor)}/#{trunc(total / divisor)}#{unit}"
  end

  defp bytes_unit_and_divisor(count) do
    cond do
      count > 1_000_000 -> {"MB", 1_000_000}
      count > 1_000 -> {"KB", 1_000}
      true -> {"B", 1}
    end
  end

  defp terminal_width do
    case :io.columns() do
      {:ok, width} -> width
      {:error, _} -> 80
    end
  end
end
