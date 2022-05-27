defmodule Bumblebee.Architecture do
  @moduledoc """
  An interface for model architectures.
  """

  @doc """
  Builds configuration specific to this model architecture.
  """
  @callback config(opts :: keyword()) :: map()

  @doc """
  The prefix used to namespace layers of the base model when used as
  part of a specialized model.

  Since the base model is a subset of more specialized models, pre-trained
  parameters from one can be applied to the other. The prefix helps to
  determine layer name mapping.
  """
  @callback base_model_prefix() :: String.t()
end
