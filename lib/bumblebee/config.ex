defmodule Bumblebee.Config do
  @moduledoc false

  @doc """
  Returns a docstring describing the given subset of common options.
  """
  @spec common_options_docs(list(atom())) :: String.t()
  def common_options_docs(keys) do
    docs = [
      output_hidden_states: """
      whether the model should return all hidden states. Defaults to `false`
      """,
      id2label: """
      a map from class index to label. Defaults to `%{}`
      """,
      num_labels: """
      the number of labels to use in the last layer for the classification \
      task. Inferred from `:id2label` if given, otherwise defaults to `2`
      """
    ]

    items =
      for {key, doc} <- docs, key in keys do
        "  * `#{inspect(key)}` - #{doc}"
      end

    """
    ## Common options

    #{Enum.join(items, "\n\n")}
    """
  end

  @doc """
  Builds model configuration.

  This function is generally used to build more a specific
  configuration for individual models. It handles configuration
  options common across multiple models specified in `common_keys`
  and extends them with additional options specified in `defaults`.

  `config_opts` can be either a keyword list with valid values or a
  map with parsed JSON data to load the configuration from.

  ## Options

    * `:atoms` - a list of configuration keys that hold atom values

  """
  @spec build_config(keyword() | map(), list(atom()), keyword(), keyword()) :: map()
  def build_config(config_opts, common_keys, defaults, opts \\ []) do
    atoms = opts[:atoms] || []

    common_config = common_config(config_opts, common_keys)

    config =
      do_build_config(config_opts, defaults, fn key, value ->
        if key in atoms and is_binary(value) do
          String.to_atom(value)
        else
          value
        end
      end)

    Map.merge(common_config, config)
  end

  defp common_config(config_opts, keys) do
    defaults = [
      id2label: %{},
      num_labels: nil,
      output_hidden_states: false
    ]

    defaults = Keyword.take(defaults, keys)

    config =
      do_build_config(config_opts, defaults, fn
        :id2label, id2label ->
          Map.new(id2label, fn {key, value} -> {String.to_integer(key), value} end)

        _key, value ->
          value
      end)

    config
    |> case do
      %{num_labels: nil} = config ->
        num_labels =
          case config[:id2label] do
            map when map != %{} -> map_size(map)
            _ -> 2
          end

        %{config | num_labels: num_labels}

      config ->
        config
    end
    |> case do
      %{id2label: id2label} = config ->
        label2id = Map.new(id2label, fn {id, label} -> {label, id} end)
        Map.put(config, :label2id, label2id)

      config ->
        config
    end
  end

  defp do_build_config(config_opts, defaults, _cast_fun) when is_list(config_opts) do
    keys = Keyword.keys(defaults)
    config_opts = Keyword.take(config_opts, keys)
    defaults |> Keyword.merge(config_opts) |> Map.new()
  end

  defp do_build_config(config_opts, defaults, cast_fun) when is_map(config_opts) do
    for {key, value} <- defaults, into: %{} do
      value =
        case Map.fetch(config_opts, Atom.to_string(key)) do
          {:ok, new_value} -> cast_fun.(key, new_value)
          :error -> value
        end

      {key, value}
    end
  end
end
