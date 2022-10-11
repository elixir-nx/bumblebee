defmodule Bumblebee.Configurable do
  @moduledoc """
  An interface for configurable entities.

  A module implementing this behaviour is expected to define a struct
  with configuration.
  """

  @type t :: struct()

  @doc """
  Configures the struct.
  """
  @callback config(t(), keyword()) :: t()
end
