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
  def map_params_source_layer_names(%{} = param_builders, fun) do
    Map.new(param_builders, fn {param_name, {sources, builder_fun}} ->
      sources =
        for ref_or_refs <- sources do
          case ref_or_refs do
            {layer_name, param_name} ->
              {fun.(layer_name), param_name}

            refs ->
              for {layer_name, param_name} <- refs, do: {fun.(layer_name), param_name}
          end
        end

      {param_name, {sources, builder_fun}}
    end)
  end

  def map_params_source_layer_names(layer_names, fun) when is_list(layer_names) do
    Enum.map(layer_names, fun)
  end

  def map_params_source_layer_names(layer_name, fun) when is_binary(layer_name) do
    fun.(layer_name)
  end
end
