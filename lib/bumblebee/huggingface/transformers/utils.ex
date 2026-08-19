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

  @doc """
  Expands every source layer name in a params mapping into a list of
  alternative names, tried in order when loading parameters.

  This is useful when the same model can be loaded from checkpoints that
  store the parameters under different names, most notably when a text
  model is a part of a larger multimodal checkpoint.
  """
  @spec expand_params_mapping_source_layer_names(
          Transformers.Model.params_mapping(),
          (String.t() -> list(String.t()))
        ) :: Transformers.Model.params_mapping()
  def expand_params_mapping_source_layer_names(params_mapping, fun) do
    Map.new(params_mapping, fn {target_layer_name, params_source} ->
      {target_layer_name, expand_params_source_layer_names(params_source, fun)}
    end)
  end

  defp expand_params_source_layer_names(%{} = param_builders, fun) do
    Map.new(param_builders, fn {param_name, {sources, builder_fun}} ->
      sources =
        for ref_or_refs <- sources do
          for {layer_name, source_param_name} <- List.wrap(ref_or_refs),
              layer_name <- fun.(layer_name),
              do: {layer_name, source_param_name}
        end

      {param_name, {sources, builder_fun}}
    end)
  end

  defp expand_params_source_layer_names(layer_names, fun) when is_list(layer_names) do
    Enum.flat_map(layer_names, fun)
  end

  defp expand_params_source_layer_names(layer_name, fun) when is_binary(layer_name) do
    fun.(layer_name)
  end
end
