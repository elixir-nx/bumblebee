defmodule Bumblebee.HuggingFace.Transformers.Utils do
  @moduledoc false

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.HuggingFace.Transformers

  @doc """
  Prefixes target and source layer names in the given params mapping.
  """
  @spec prefix_params_mapping(
          Transformers.Model.params_mapping(),
          String.t() | nil,
          String.t() | nil
        ) :: Transformers.Model.params_mapping()
  def prefix_params_mapping(params_mapping, target_prefix, source_prefix) do
    Map.new(params_mapping, fn {target_layer_name, params_source} ->
      {
        join(target_prefix, target_layer_name),
        map_params_source_layer_names(params_source, &join(source_prefix, &1))
      }
    end)
  end

  @doc """
  Maps layer names in a params mapping value.
  """
  @spec map_params_source_layer_names(
          Transformers.Model.params_source(),
          (String.t() -> String.t())
        ) :: Transformers.Model.params_source()
  def map_params_source_layer_names(%{} = params_source, fun) do
    Map.new(params_source, fn {param_name, {sources, source_fun}} ->
      sources = for {layer_name, param_name} <- sources, do: {fun.(layer_name), param_name}
      {param_name, {sources, source_fun}}
    end)
  end

  def map_params_source_layer_names(layer_name, fun) when is_binary(layer_name) do
    fun.(layer_name)
  end
end
