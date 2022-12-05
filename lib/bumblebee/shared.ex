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
    defaults = Keyword.validate!(defaults, forced_bos_token_id: nil, forced_eos_token_id: nil)

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
  @spec common_options_from_transformers(map(), Bumblebee.ModelSpec.t()) :: keyword()
  def common_options_from_transformers(data, spec) do
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
      forced_bos_token_id: {"forced_bos_token_id", number()},
      forced_eos_token_id: {"forced_eos_token_id", number()}
    ]

    converters =
      Keyword.filter(converters, fn {key, _} ->
        Map.has_key?(spec, key)
      end)

    opts = convert!(data, converters)

    if Map.has_key?(spec, :num_labels) and
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
  @spec validate_label_options(Bumblebee.ModelSpec.t()) :: Bumblebee.ModelSpec.t()
  def validate_label_options(%{num_labels: num_labels, id_to_label: id_to_label} = spec) do
    if id_to_label != %{} and map_size(id_to_label) != spec.num_labels do
      raise ArgumentError,
            "size mismatch between :num_labels (#{inspect(num_labels)}) and :id_to_label (#{inspect(id_to_label)})"
    end

    spec
  end

  @doc """
  Optionally unwraps a singular list.
  """
  @spec normalize_output(list(), boolean()) :: list(term()) | term()
  def normalize_output(list, multi?)

  def normalize_output([term], false), do: term
  def normalize_output(list, true), do: list

  @doc """
  Validates and normalizes task input.
  """
  @spec validate_serving_input!(term(), (term() -> boolean()), String.t()) ::
          {list(term()), multi? :: boolean()}
  def validate_serving_input!(input, validator, description) do
    cond do
      validator.(input) ->
        {[input], false}

      is_list(input) ->
        for item <- input do
          unless validator.(item) do
            raise ArgumentError,
                  "expected each input in the input list to be #{description}, found: #{inspect(item)}"
          end
        end

        {input, true}

      true ->
        raise ArgumentError, "expected the input to be #{description}, got: #{inspect(input)}"
    end
  end

  @doc """
  Asserts that the model architecture matches one of the expected
  architectures.
  """
  def validate_architecture!(spec, architecture)

  def validate_architecture!(spec, architectures) when is_list(architectures) do
    unless spec.architecture in architectures do
      raise ArgumentError,
            "expected a model architecture to be either of #{inspect(architectures)}, got #{inspect(spec.architecture)}"
    end
  end

  def validate_architecture!(spec, architecture) do
    unless spec.architecture == architecture do
      raise ArgumentError,
            "expected a model with architecture #{inspect(architecture)}, got #{inspect(spec.architecture)}"
    end
  end

  @doc """
  Checks if the given term is an image.
  """
  @spec image?(term()) :: boolean()
  def image?(image) do
    try do
      Nx.to_template(image)
    rescue
      Protocol.UndefinedError -> false
    else
      %Nx.Tensor{shape: {_, _, channels}} when channels in 1..4 -> true
      _ -> false
    end
  end

  @doc """
  Pads a batch to the given size, if given.

  When the batch exceeds `batch_size`, raises an error.
  """
  @spec maybe_pad(Nx.Batch.t(), non_neg_integer() | nil) :: Nx.Batch.t()
  def maybe_pad(batch, batch_size)

  def maybe_pad(batch, nil), do: batch

  def maybe_pad(%{size: size}, batch_size) when size > batch_size do
    raise ArgumentError,
          "input batch size (#{size}) exceeds the maximum configured batch size (#{batch_size})"
  end

  def maybe_pad(%{size: size} = batch, batch_size) do
    Nx.Batch.pad(batch, batch_size - size)
  end

  @doc """
  Compiles or wraps the function with just-in-time compilation.

  When `compile?` is `true`, runs `template_fun` to get template args
  and calls compiles the function upfront. The template function may
  return a mix of tensors and templates, all arguments are automatically
  converter to templates.
  """
  @spec compile_or_jit(
          function(),
          keyword(),
          boolean(),
          (() -> list(Nx.Tensor.t()))
        ) :: function()
  def compile_or_jit(fun, defn_options, compile?, template_fun) do
    if compile? do
      template_args = template_fun.() |> templates()
      Nx.Defn.compile(fun, template_args, defn_options)
    else
      Nx.Defn.jit(fun, defn_options)
    end
  end

  @doc """
  Returns at template for the given model input.

  Replaces leading axis sizes with `overrides`.
  """
  @spec input_template(
          Bumblebee.ModelSpec.t(),
          String.t(),
          list(non_neg_integer())
        ) :: Nx.Tensor.t()
  def input_template(%module{} = spec, name, overrides) do
    %{^name => template} = module.input_template(spec)

    shape =
      overrides
      |> Enum.with_index()
      |> Enum.reduce(Nx.shape(template), fn {size, idx}, shape ->
        put_elem(shape, idx, size)
      end)

    Nx.template(shape, Nx.type(template))
  end

  @doc """
  Converts tensors to templates.
  """
  @spec templates(list(Nx.Tensor.t())) :: list(Nx.Tensor.t())
  def templates(list) do
    Enum.map(list, fn
      %Nx.Tensor{data: %Nx.TemplateBackend{}} = template -> template
      other -> Nx.to_template(other)
    end)
  end
end
