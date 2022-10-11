defprotocol Bumblebee.HuggingFace.Transformers.Config do
  @moduledoc false

  # This protocol defines a bridge between Bumblebee and huggingface/transformers
  # configuration.

  @doc """
  Updates configuration based on a parsed JSON data.
  """
  @spec load(t(), map()) :: Bumblebee.ModelSpec.t()
  def load(config, data)
end
