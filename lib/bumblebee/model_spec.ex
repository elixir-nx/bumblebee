defmodule Bumblebee.ModelSpec do
  @moduledoc """
  An interface for configuring and building models based on the same
  architecture.

  Every module implementing this behaviour is expected to also define
  a configuration struct.
  """

  @typedoc """
  Model configuration and metadata.
  """
  @type t :: %{
          optional(atom()) => term(),
          __struct__: atom(),
          architecture: atom()
        }

  @doc """
  Returns the list of supported model architectures.
  """
  @callback architectures :: list(atom())

  @doc """
  Returns the prefix used to namespace layers of the base model when
  used as part of a specialized model.

  Since the base model is a subset of more specialized models, pre-trained
  parameters from one can be applied to the other. The prefix helps
  to determine layer name mapping.
  """
  @callback base_model_prefix() :: String.t()

  @doc """
  Configures the model.
  """
  @callback config(t(), keyword()) :: t()

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
