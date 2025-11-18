defmodule Bumblebee.LogitsProcessor do
  @moduledoc """
  An interface for configuring and using logits processors.

  Logits processors are used during autoregressive generation to modify
  predicted scores at each generation step. This allows for applying
  certain rules to the model output to control which tokens are picked
  at each generation step, and which are not.

  Every module implementing this behaviour is expected to also define
  a configuration struct.
  """

  @type t :: Bumblebee.Configurable.t()

  @type state :: Nx.Container.t()

  @type process_context :: %{
          sequence: Nx.Tensor.t(),
          length: Nx.Tensor.t(),
          input_length: Nx.Tensor.t()
        }

  @type init_context :: %{}

  @doc """
  Initializes state for a new logits processor.

  Returns `state`, which is an opaque `Nx.Container`, and it is then
  passed to and returned from `process/2`.

  Oftentimes logits processors are stateless, in which case this
  function can return an empty container, such as `{}`.
  """
  @callback init(t(), init_context()) :: state()

  @doc """
  Processes logits, applying specific rules.
  """
  @callback process(
              t(),
              state(),
              logits :: Nx.Tensor.t(),
              context :: process_context()
            ) :: {state :: map(), logits :: Nx.Tensor.t()}
end
