defmodule Bumblebee.Utils.Model do
  @moduledoc false

  @doc """
  Builds a model output based on all outputs and configuration.

  Skips outputs that are excluded via model configuration.
  """
  @spec output(map(), Bumblebee.ModelSpec.t()) :: map()
  def output(outputs, config) do
    output_hidden_states = Map.get(config, :output_hidden_states, true)
    output_attentions = Map.get(config, :output_attentions, true)
    use_cache = Map.get(config, :use_cache, true)

    outputs
    |> drop_falsy(
      hidden_states: output_hidden_states,
      encoder_hidden_states: output_hidden_states,
      decoder_hidden_states: output_hidden_states,
      attentions: output_attentions,
      encoder_attentions: output_attentions,
      decoder_attentions: output_attentions,
      cross_attentions: output_attentions,
      cache: use_cache
    )
    |> Axon.container()
  end

  defp drop_falsy(outputs, filters) do
    for {key, include?} <- filters, not include?, reduce: outputs do
      outputs -> Map.delete(outputs, key)
    end
  end

  @doc """
  Adds another word to a hierarchical name.
  """
  @spec join(String.t() | nil, String.Chars.t()) :: String.t()
  def join(name, suffix)

  def join(nil, suffix), do: to_string(suffix)
  def join(name, suffix), do: name <> "." <> to_string(suffix)
end
