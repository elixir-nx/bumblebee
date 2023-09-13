defmodule Bumblebee.Featurizer do
  @moduledoc """
  An interface for configuring and applying featurizers.

  A featurizer is used to convert raw data into model input.

  Every module implementing this behaviour is expected to also define
  a configuration struct.
  """

  @type t :: Bumblebee.Configurable.t()

  @doc """
  Converts the given input to a batched tensor (or a tensor container).

  Numerical batch processing should be moved to `c:process_batch/2`
  whenever possible.
  """
  @callback process_input(t(), input :: any()) :: Nx.t() | Nx.Container.t()

  @doc """
  Returns an input template for `c:process_batch/2`.

  The shape is effectively the same as the result of `c:process_input/2`,
  except for the batch size.
  """
  @callback batch_template(t(), batch_size :: pos_integer()) :: Nx.t() | Nx.Container.t()

  @doc """
  Optional batch processing stage.

  This is a numerical function. It receives the result of `c:process_input/2`,
  except the batch size may differ.

  When using featurizer as part of `Nx.Serving`, the batch stage can
  be merged with the model computation and compiled together.
  """
  @callback process_batch(t(), input :: Nx.t() | Nx.Container.t()) :: Nx.t() | Nx.Container.t()

  @optional_callbacks batch_template: 2, process_batch: 2

  @doc """
  Converts the given input to a batched tensor (or a tensor container).
  """
  @spec process_input(t(), any()) :: Nx.t() | Nx.Container.t()
  def process_input(%module{} = featurizer, input) do
    module.process_input(featurizer, input)
  end

  @doc """
  Returns an input template for `process_batch/2`.

  If the featurizer does not define batch processing, `nil` is returned.
  """
  @spec batch_template(t(), pos_integer()) :: Nx.t() | Nx.Container.t() | nil
  def batch_template(%module{} = featurizer, batch_size) do
    if Code.ensure_loaded?(module) and function_exported?(module, :batch_template, 2) do
      module.batch_template(featurizer, batch_size)
    end
  end

  @doc """
  Optional batch processing stage.

  This is a numerical function. It receives the result of `c:process_input/2`,
  except the batch size may differ.

  If the featurizer does not define batch processing, the input is
  returned as is.
  """
  @spec process_batch(t(), Nx.t() | Nx.Container.t()) :: Nx.t() | Nx.Container.t()
  def process_batch(%module{} = featurizer, batch) do
    if Code.ensure_loaded?(module) and function_exported?(module, :process_batch, 2) do
      module.process_batch(featurizer, batch)
    else
      batch
    end
  end
end
