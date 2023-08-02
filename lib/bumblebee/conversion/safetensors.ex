defmodule Bumblebee.Conversion.Safetensors do
  @doc """
  Loads data saved as [Safetensors](https://huggingface.co/docs/safetensors).

  They are automatically deserialized as Nx tensors.
  """
  @spec load!(Path.t()) :: term()
  def load!(path) do
    path
    |> File.read!()
    |> Safetensors.load!()
  end
end
