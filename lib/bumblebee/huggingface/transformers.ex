defprotocol Bumblebee.HuggingFace.Transformers.Config do
  @moduledoc """
  This protocol defines a bridge between bumblebee and huggingface/transformers
  model configuration.
  """

  @doc """
  Returns a map with model architectures as keys and the corresponding
  huggingface/transformers classes as values.
  """
  @spec architecture_mapping(t()) :: %{atom() => String.t()}
  def architecture_mapping(config)

  @doc """
  Updates model configuration based on a parsed JSON data.
  """
  @spec load(t(), map()) :: Bumblebee.ModelSpec.t()
  def load(config, data)
end
