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
  Generates documentation string for the given options specification.
  """
  @spec options_doc(keyword()) :: String.t()
  def options_doc(options) do
    items =
      for {key, info} <- options, doc = info[:doc] do
        doc = String.replace(doc, "\n", "\n    ")
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
      decoder_start_token_id: {"decoder_start_token_id", number()}
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
  @spec validate_serving_input!(
          term(),
          (term() -> {:ok, term()} | {:error, String.t()})
        ) :: {list(term()), multi? :: boolean()}
  def validate_serving_input!(input, validator)

  def validate_serving_input!(input, validator) when is_list(input) do
    input =
      for item <- input do
        case validator.(item) do
          {:ok, normalized} -> normalized
          {:error, message} -> raise ArgumentError, "invalid input in the batch, #{message}"
        end
      end

    {input, true}
  end

  def validate_serving_input!(input, validator) do
    case validator.(input) do
      {:ok, normalized} -> {[normalized], false}
      {:error, message} -> raise ArgumentError, "invalid input, #{message}"
    end
  end

  def validate_image(input) do
    if image?(input) do
      {:ok, input}
    else
      {:error, "expected an image, got: #{inspect(input)}"}
    end
  end

  def validate_string(input) do
    if is_binary(input) do
      {:ok, input}
    else
      {:error, "expected a string, got: #{inspect(input)}"}
    end
  end

  @doc """
  Validates that the input is a single value and not a batch.
  """
  @spec validate_input_for_stream!(term()) :: :ok
  def validate_input_for_stream!(input) do
    if is_list(input) do
      raise ArgumentError,
            "serving only accepts singular input when stream is enabled," <>
              " call the serving with each input in the batch separately"
    end

    :ok
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
  Asserts that the given options keyword list has all of the given
  keys.
  """
  def require_options!(opts, keys) do
    missing = keys -- Keyword.keys(opts)

    if missing != [] do
      raise ArgumentError, "missing keys #{inspect(missing)} in #{inspect(opts)}"
    end

    opts
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
  Shared logic applied after serving computation to the resulting tensor
  or container.
  """
  @spec serving_post_computation(result) :: result when result: Nx.Tensor.t() | Nx.Container.t()
  def serving_post_computation(result) do
    # We transfer to binary backend so tensor access in post-processing
    # is not blocked by the serving the serving computation. It is also
    # necessary when partitions are enabled since we may need to
    # concatenate results for input exceeding the expected batch size.
    Nx.backend_transfer(result, Nx.BinaryBackend)
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
          (-> list(Nx.Tensor.t()))
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

  @doc """
  Converts logits to scores as per the given scores function.

  Raises `ArgumentError` if the scores function is invalid.
  """
  @spec logits_to_scores(Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def logits_to_scores(logits, scores_function) do
    case scores_function do
      :softmax ->
        Axon.Activations.softmax(logits)

      :sigmoid ->
        Axon.Activations.sigmoid(logits)

      :none ->
        logits

      other ->
        raise ArgumentError,
              "expected :scores_function to be either of :softmax, :sigmoid or :none, got: #{inspect(other)}"
    end
  end

  @doc """
  Returns batch keys for the given sequence length specified in text
  serving compile options.
  """
  @spec sequence_batch_keys(nil | non_neg_integer() | list(non_neg_integer())) :: list()
  def sequence_batch_keys(sequence_length)

  def sequence_batch_keys(nil), do: [:default]

  def sequence_batch_keys(length) when is_number(length) do
    [{:sequence_length, length}]
  end

  def sequence_batch_keys(lengths) when is_list(lengths) do
    Enum.map(lengths, &{:sequence_length, &1})
  end

  @doc """
  Determines batch key compatible with `sequence_batch_keys/1` based
  on tokenized inputs.
  """
  @spec sequence_batch_key_for_inputs(
          inputs :: any(),
          nil | non_neg_integer() | list(non_neg_integer())
        ) :: term()
  def sequence_batch_key_for_inputs(inputs, sequence_length) do
    if sequence_length do
      {:sequence_length, Nx.axis_size(inputs["input_ids"], 1)}
    else
      :default
    end
  end

  @doc """
  If `preallocate?` is `true`, allocates `params` using `defn_options`.
  """
  @spec maybe_preallocate(map(), boolean(), keyword()) :: map()
  def maybe_preallocate(params, preallocate?, defn_options) do
    if preallocate? do
      backend = Nx.Defn.to_backend(defn_options)
      Nx.backend_copy(params, backend)
    else
      params
    end
  end

  @doc """
  Slices a subset of dense layer parameters.

  Expects `out_template` to be a tuple representing a "shape" of the
  output units. The tuple should include a list in place of the axis
  along which the parameters are concatenated. The list should contain
  chunk sizes. `chunk_idx` indicates which chunk to slice.
  """
  def sliced_dense_params_source(source_layer_name, out_template, chunk_idx) do
    out_template = Tuple.to_list(out_template)
    chunk_axis = Enum.find_index(out_template, &is_list/1)
    chunk_sizes = Enum.at(out_template, chunk_axis)
    {prev_chunk_sizes, [chunk_size | _]} = Enum.split(chunk_sizes, chunk_idx)
    offset = Enum.sum(prev_chunk_sizes)
    out_shape = List.replace_at(out_template, chunk_axis, Enum.sum(chunk_sizes))

    %{
      "kernel" => {
        [{source_layer_name, "weight"}],
        fn [kernel] ->
          in_size = Nx.axis_size(kernel, -1)

          kernel =
            kernel
            |> Nx.reshape(List.to_tuple(out_shape ++ [in_size]))
            |> Nx.slice_along_axis(offset, chunk_size, axis: chunk_axis)
            |> Nx.reshape({:auto, in_size})

          # Transpose the kernel
          [out_features, in_features] = Nx.axes(kernel)
          Nx.transpose(kernel, axes: [in_features, out_features])
        end
      },
      "bias" => {
        [{source_layer_name, "bias"}],
        fn [bias] ->
          bias
          |> Nx.reshape(List.to_tuple(out_shape))
          |> Nx.slice_along_axis(offset, chunk_size, axis: chunk_axis)
          |> Nx.flatten()
        end
      }
    }
  end

  @type featurizer_image_size ::
          %{height: non_neg_integer(), width: non_neg_integer()}
          | %{shortest_edge: non_neg_integer()}

  @doc """
  Returns an exact `{height, width}` size to resize images into.

  Accepts a featurizer size map.
  """
  @spec featurizer_resize_size(Nx.Tensor.t(), featurizer_image_size()) ::
          {height :: non_neg_integer(), width :: non_neg_integer()}
  def featurizer_resize_size(images, size)

  def featurizer_resize_size(_images, %{height: height, width: width}), do: {height, width}

  def featurizer_resize_size(images, %{shortest_edge: size}) do
    {height, width} = images_spacial_sizes(images)

    {short, long} = if height < width, do: {height, width}, else: {width, height}

    out_short = size
    out_long = floor(size * long / short)

    if height < width, do: {out_short, out_long}, else: {out_long, out_short}
  end

  defp images_spacial_sizes(images) do
    height = Nx.axis_size(images, -3)
    width = Nx.axis_size(images, -2)
    {height, width}
  end

  @doc """
  Checks whether if the given featurizer image size is fixed or depends
  on the input size.
  """
  @spec featurizer_size_fixed?(featurizer_image_size()) :: boolean()
  def featurizer_size_fixed?(size)

  def featurizer_size_fixed?(%{height: _, width: _}), do: true
  def featurizer_size_fixed?(%{shortest_edge: _}), do: false
end
