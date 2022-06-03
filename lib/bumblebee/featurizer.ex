defmodule Bumblebee.Featurizer do
  @moduledoc """
  An interface for configuring and applying featurizers.

  A featurizer is used to convert raw data into model input.

  Every module implementing this behaviour is expected to also define
  a configuration struct.
  """

  @typedoc """
  Featurizer configuration and metadata.
  """
  @type t :: %{
          optional(atom()) => term(),
          __struct__: atom()
        }

  @doc """
  Configures the featurizer.
  """
  @callback config(t(), keyword()) :: t()

  @doc """
  Performs feature extraction on the given input.
  """
  @callback apply(t(), any()) :: any()
end
