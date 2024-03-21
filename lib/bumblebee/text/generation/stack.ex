defmodule Bumblebee.Text.Generation.Stack do
  @moduledoc false

  # A "stack" like data structure represented as an Nx container
  # to make constrained sampling possible/easier. The HF implementation
  # uses a "dynamic" stack, but we need all shapes up front and
  # can't manipulate so we use a "stack" and then a pointer in
  # the stack

  alias __MODULE__

  @derive {Nx.Container, containers: [:data, :pointer]}
  defstruct [:data, :pointer]

  import Nx.Defn

  @empty_value -1

  @doc """
  Initializes a new stack.
  """
  def new(size, opts \\ []) do
    opts = Keyword.validate!(opts, type: :s64)

    %Stack{
      data: Nx.broadcast(Nx.tensor(@empty_value, type: opts[:type]), {size}),
      pointer: Nx.tensor(0)
    }
  end

  @doc """
  Push a value to the top of the stack.
  """
  deftransform push(%Stack{data: data, pointer: pointer} = stack, value) do
    unless Nx.rank(value) == 0, do: raise("can only push scalar values to stack")

    %{
      data: Nx.put_slice(data, [pointer], value),
      pointer: Nx.add(pointer, 1)
    }
  end

  @doc """
  Pops a value from the stack.
  """
  defn pop(%Stack{data: data, pointer: pointer} = stack) do
    value = data[[pointer]]
    {value, %{stack | pointer: Nx.subtract(pointer, 1)}}
  end

  @doc """
  Peeks at the head of the stack.
  """
  defn peek(%Stack{data: data, pointer: pointer}) do
    data[[pointer]]
  end

  @doc """
  Returns the length of the stack.
  """
  defn length(%Stack{pointer: pointer}) do
    pointer
  end
end
