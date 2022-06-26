defmodule Bumblebee.Shared do
  @moduledoc false

  import Nx.Defn

  @pi 3.14159265358979323

  @doc """
  Returns a subset of common config attributes with their default
  values.
  """
  @spec common_config_defaults(list(atom())) :: keyword()
  def common_config_defaults(keys) do
    [
      output_hidden_states: false,
      output_attentions: false,
      id2label: %{},
      label2id: %{},
      num_labels: 2
    ]
    |> Keyword.take(keys)
  end

  @doc """
  Returns a docstring describing the given subset of common config
  attributes.
  """
  @spec common_config_docs(list(atom())) :: String.t()
  def common_config_docs(keys) do
    docs = [
      output_hidden_states: """
      whether the model should return all hidden states. Defaults to `false`
      """,
      output_attentions: """
      whether the model should return all attentions. Defaults to `false`
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
    #{Enum.join(items, "\n\n")}
    """
  end

  @doc """
  Merges the given list of attributes into a configuration struct.

  Attributes that are not present in the configuration struct are
  ignored.
  """
  @spec put_config_attrs(struct(), keyword()) :: struct()
  def put_config_attrs(config, opts) do
    Enum.reduce(opts, config, fn {key, value}, config ->
      case config do
        %{^key => _} -> %{config | key => value}
        _ -> config
      end
    end)
  end

  # @doc """
  # Sets common configuration attributes that are computed based on
  # other attributes.
  # """
  @spec add_common_computed_options(keyword()) :: keyword()
  def add_common_computed_options(opts) do
    opts =
      case {opts[:num_labels], opts[:id2label]} do
        {nil, %{} = id2label} when id2label != %{} ->
          put_in(opts[:num_labels], map_size(id2label))

        {nil, _id2label} ->
          opts

        {_num_labels, nil} ->
          put_in(opts[:id2label], %{})

        {num_labels, id2label} when map_size(id2label) not in [0, num_labels] ->
          raise ArgumentError,
                "size mismatch between :num_labels (#{inspect(num_labels)}) and :id2label (#{inspect(id2label)})"

        _ ->
          opts
      end

    if id2label = opts[:id2label] do
      label2id = Map.new(id2label, fn {id, label} -> {label, id} end)
      put_in(opts[:label2id], label2id)
    else
      opts
    end
  end

  @doc """
  Converts values for the given keys to atoms.
  """
  @spec atomize_values(map(), list(String.t())) :: map()
  def atomize_values(data, keys) do
    Enum.reduce(keys, data, fn key, data ->
      update(data, key, fn
        nil -> nil
        string -> String.to_atom(string)
      end)
    end)
  end

  @doc """
  Transforms values for common options wherever necessary.
  """
  @spec cast_common_values(map()) :: map()
  def cast_common_values(data) do
    update(data, "id2label", fn id2label ->
      Map.new(id2label, fn {key, value} -> {String.to_integer(key), value} end)
    end)
  end

  defp update(data, key, fun) do
    case data do
      %{^key => value} -> %{data | key => fun.(value)}
      data -> data
    end
  end

  @doc """
  Loads the given parsed JSON data into a configuration struct.
  """
  @spec data_into_config(map(), struct()) :: struct()
  def data_into_config(data, %module{} = config) do
    opts =
      Enum.reduce(Map.keys(config) -- [:architecture], [], fn key, opts ->
        case Map.fetch(data, Atom.to_string(key)) do
          {:ok, value} -> [{key, value} | opts]
          :error -> opts
        end
      end)

    module.config(config, opts)
  end

  @doc """
  Maps activations to known axon activations.
  """
  @spec map_activations(map()) :: map()
  def map_activations(data) do
    activation_keys = ["hidden_act"]

    Enum.reduce(activation_keys, data, fn key, data ->
      Map.replace(data, key, axon_activation(data[key]))
    end)
  end

  defp axon_activation(:gelu_new), do: &gelu_new/2
  defp axon_activation(activation), do: activation

  # TODO: We should add pi to Nx
  defnp gelu_new(input, _opts \\ []) do
    0.5 * input *
      (1.0 + Nx.tanh(Nx.sqrt(2.0 / @pi) * (input + 0.044715 * Nx.power(input, 3.0))))
  end
end
