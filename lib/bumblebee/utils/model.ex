defmodule Bumblebee.Utils.Model do
  @moduledoc false

  @doc """
  Adds another word to a hierarchical name.
  """
  @spec join(String.t() | nil, String.Chars.t()) :: String.t()
  def join(name, suffix)

  def join(nil, suffix), do: to_string(suffix)
  def join(name, suffix), do: name <> "." <> to_string(suffix)

  @doc """
  Converts a list of inputs to a map with input names as keys.
  """
  @spec inputs_to_map(list(Axon.t())) :: %{String.t() => Axon.t()}
  def inputs_to_map(inputs) when is_list(inputs) do
    for %Axon{output: id, nodes: nodes} = axon <- inputs, into: %{} do
      input = nodes[id]
      {input.name.(:input, %{}), axon}
    end
  end
end
