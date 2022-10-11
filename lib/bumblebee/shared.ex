defmodule Bumblebee.Shared do
  @moduledoc false

  @doc """
  Returns specification for the given common options.
  """
  @spec common_options(list(atom())) :: keyword()
  def common_options(keys) do
    common_options = [
      output_hidden_states: [
        default: false,
        doc: "whether the model should return all hidden states"
      ],
      output_attentions: [
        default: false,
        doc: "whether the model should return all attentions"
      ],
      num_labels: [
        default: 2,
        doc: "the number of labels to use in the last layer for the classification task"
      ],
      id_to_label: [
        default: %{},
        doc: "a map from class index to label"
      ],
      use_cross_attention: [
        default: false,
        doc:
          "whether cross-attention layers should be added to the model." <>
            "This is only relevant for decoder models"
      ]
    ]

    Keyword.take(common_options, keys)
  end

  @doc """
  Returns specification for the token options with the corresponding
  defaults.
  """
  @spec token_options(keyword()) :: keyword()
  def token_options(defaults) do
    for {key, default} <- defaults do
      {key, [default: default, doc: nil]}
    end
  end

  @doc """
  Returns options for generation.

  Default option values can be overridden through `defaults`.
  """
  @spec generation_options(keyword()) :: keyword()
  def generation_options(defaults \\ []) do
    defaults =
      Keyword.validate!(defaults,
        min_length: 0,
        max_length: 20,
        forced_bos_token_id: nil,
        forced_eos_token_id: nil
      )

    for {key, default} <- defaults do
      {key, [default: default, doc: nil]}
    end
  end

  @doc """
  Generates documentation string for the given options specification.
  """
  @spec options_doc(keyword()) :: String.t()
  def options_doc(options) do
    items =
      for {key, info} <- options, doc = info[:doc] do
        item = "  * `#{inspect(key)}` - #{doc}"

        case info[:default] do
          nil -> item
          default -> "#{item}. Defaults to `#{inspect(default)}`"
        end
      end

    Enum.join(items, "\n\n")
  end

  @doc """
  Returns option defaults form the options specification.

  This function is useful in combination with `defstruct`.
  """
  @spec option_defaults(keyword()) :: keyword()
  def option_defaults(options) do
    for {key, info} <- options, do: {key, info[:default]}
  end

  @doc """
  Converts common options from huggingface/transformers configuration.
  """
  @spec common_options_from_transformers(map(), struct()) :: keyword()
  def common_options_from_transformers(data, config) do
    import Bumblebee.Shared.Converters

    converters = [
      output_hidden_states: {"output_hidden_states", boolean()},
      output_attentions: {"output_attentions", boolean()},
      num_labels: {"num_labels", number()},
      id_to_label: {"id2label", map(integer_as_string(), string())},
      use_cross_attention: {"use_cross_attention", false},
      # Tokens
      pad_token_id: {"pad_token_id", number()},
      bos_token_id: {"bos_token_id", number()},
      eos_token_id: {"eos_token_id", number()},
      decoder_start_token_id: {"decoder_start_token_id", number()},
      # Generation
      min_length: {"min_length", number()},
      max_length: {"max_length", number()},
      forced_bos_token_id: {"forced_bos_token_id", number()},
      forced_eos_token_id: {"forced_eos_token_id", number()}
    ]

    converters =
      Keyword.filter(converters, fn {key, _} ->
        Map.has_key?(config, key)
      end)

    opts = convert!(data, converters)

    if Map.has_key?(config, :num_labels) and
         not Keyword.has_key?(opts, :num_labels) and opts[:id_to_label] do
      Keyword.put(opts, :num_labels, map_size(opts[:id_to_label]))
    else
      opts
    end
  end

  @doc """
  Merges the given list of attributes into a configuration struct.

  Raises `ArgumentError` if an invalid attribute name is found.
  """
  @spec put_config_attrs(struct(), keyword()) :: struct()
  def put_config_attrs(config, opts) do
    Enum.reduce(opts, config, fn {key, value}, config ->
      case config do
        %{^key => _} ->
          %{config | key => value}

        _ ->
          raise ArgumentError,
                "unexpected attribute #{inspect(key)} for %#{inspect(config.__struct__)}{}"
      end
    end)
  end

  @doc """
  Validates that label-related attributes have consistent size.
  """
  @spec validate_label_options(struct()) :: struct()
  def validate_label_options(%{num_labels: num_labels, id_to_label: id_to_label} = config) do
    if id_to_label != %{} and map_size(id_to_label) != config.num_labels do
      raise ArgumentError,
            "size mismatch between :num_labels (#{inspect(num_labels)}) and :id_to_label (#{inspect(id_to_label)})"
    end

    config
  end
end
