defmodule Bumblebee.ModelSpec do
  @moduledoc """
  An interface for configuring and building models based on the same
  architecture.

  Every module implementing this behaviour is expected to also define
  a configuration struct.
  """

  @type t :: Bumblebee.Configurable.t()

  @doc """
  Returns the list of supported model architectures.
  """
  @callback architectures :: list(atom())

  @doc """
  Builds a template input for the model.

  The template is used to compile the model when initializing parameters.
  """
  @callback input_template(t()) :: map()

  @doc """
  Builds an `Axon` model according to the given configuration.
  """
  @callback model(t()) :: Axon.t()
end
