defmodule Bumblebee.Utils do
  @moduledoc false

  @doc """
  Checks if the progress bar is enabled globally.
  """
  @spec progress_bar_enabled? :: boolean()
  def progress_bar_enabled?() do
    Application.get_env(:bumblebee, :progress_bar_enabled, true)
  end

  @doc """
  Returns the progress bar update step in percent.

  Progress updates only when crossing step boundaries (e.g., every 1%).
  Defaults to `1`. Set to `nil` for updates on every chunk.
  """
  @spec progress_bar_step :: non_neg_integer() | nil
  def progress_bar_step() do
    Application.get_env(:bumblebee, :progress_bar_step, 1)
  end
end
