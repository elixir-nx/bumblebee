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
          "whether cross-attention layers should be added to the model. " <>
            "This is only relevant for decoder models"
      ],
      rotary_embedding_scaling_strategy: [
        default: nil,
        doc: """
        scaling configuration for rotary embedding. Currently the supported values are:

          * `%{type: :linear, factor: number()}`

          * `%{type: :dynamic, factor: number()}`

          * `%{type: :yarn, factor: number(), original_max_positions: pos_integer(), beta_fast: number(), beta_slow: number()}`

          * `%{type: :llama3, factor: number(), low_frequency_factor: number(), high_frequency_factor: number(), original_max_positions: pos_integer()}`

          * `%{type: :longrope, short_factor: list(number()), long_factor: list(number()), original_max_positions: pos_integer()}`

        YaRN also accepts `:attention_factor` (defaults to `0.1 * log(factor) + 1.0`)
        and `:truncate` (defaults to `true`).

        For more details see https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
        """
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
  Generates documentation string for the given global layer options.
  """
  @spec global_layer_options_doc(list(atom())) :: String.t()
  def global_layer_options_doc(names) do
    docs = [
      output_hidden_states: "when `true`, the model output includes all hidden states",
      output_attentions: "when `true`, the model output includes all attention weights"
    ]

    Enum.map_join(names, "\n\n", fn name ->
      doc = Keyword.fetch!(docs, name)
      "  * `#{inspect(name)}` - #{doc}"
    end)
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
  Converts the causal-attention setting from Hugging Face config data.

  Hugging Face models use `"is_causal"`, while some embedding checkpoints
  serialize `"use_bidirectional_attention"`. `"is_causal"` takes precedence
  when both are present.
  """
  @spec bidirectional_attention_options_from_transformers(map()) :: keyword()
  def bidirectional_attention_options_from_transformers(data) do
    case Map.fetch(data, "is_causal") do
      {:ok, is_causal} when is_boolean(is_causal) ->
        [use_bidirectional_attention: not is_causal]

      {:ok, _value} ->
        raise "conversion failed, expected \"is_causal\" to be a boolean"

      :error ->
        case Map.fetch(data, "use_bidirectional_attention") do
          {:ok, use_bidirectional_attention} when is_boolean(use_bidirectional_attention) ->
            [use_bidirectional_attention: use_bidirectional_attention]

          {:ok, _value} ->
            raise "conversion failed, expected \"use_bidirectional_attention\" to be a boolean"

          :error ->
            []
        end
    end
  end

  @doc false
  @spec validate_bidirectional_attention(Bumblebee.ModelSpec.t()) :: Bumblebee.ModelSpec.t()
  def validate_bidirectional_attention(%{use_bidirectional_attention: value} = spec)
      when is_boolean(value),
      do: spec

  def validate_bidirectional_attention(%{use_bidirectional_attention: value}) do
    raise ArgumentError,
          ":use_bidirectional_attention must be a boolean, got: #{inspect(value)}"
  end

  @doc """
  Converts rotary embedding options from Hugging Face config data.

  Supports both `"rope_parameters"` and the older `"rope_scaling"`
  with top-level `"rope_theta"`.

  When the parameters are given for each layer type, the options for
  `"sliding_attention"` get the `_local` suffix.
  """
  @spec rotary_embedding_options_from_transformers(map()) :: keyword()
  def rotary_embedding_options_from_transformers(data) do
    case data["rope_parameters"] || data["rope_scaling"] || %{} do
      %{"full_attention" => full_params, "sliding_attention" => sliding_params} ->
        local_opts =
          for {key, value} <- rotary_embedding_options(sliding_params, data) do
            {:"#{key}_local", value}
          end

        rotary_embedding_options(full_params, data) ++ local_opts

      params ->
        params = Map.merge(Map.take(data, ["rope_theta", "partial_rotary_factor"]), params)
        rotary_embedding_options(params, data)
    end
  end

  defp rotary_embedding_options(params, data) do
    base = params["rope_theta"]

    if base != nil and (not is_number(base) or base <= 0) do
      raise "conversion failed, \"rope_theta\" must be a positive number"
    end

    [
      rotary_embedding_base: base,
      rotary_embedding_percentage: params["partial_rotary_factor"],
      rotary_embedding_scaling_strategy: rotary_embedding_scaling_strategy(params, data)
    ]
    |> Enum.reject(fn {_key, value} -> value == nil end)
  end

  defp rotary_embedding_scaling_strategy(params, data) do
    # Phi-3 has this option at the top level, other models in the parameters
    original_max_positions =
      data["original_max_position_embeddings"] || params["original_max_position_embeddings"] ||
        data["max_position_embeddings"]

    type = rope_type!(params)

    if type == "default" and scaling_fields?(params) do
      if Map.has_key?(params, "type") or Map.has_key?(params, "rope_type") do
        raise "conversion failed, default rotary embedding does not accept scaling parameters"
      else
        raise "conversion failed, rotary embedding scaling parameters require \"type\" or \"rope_type\""
      end
    end

    case {type, params} do
      {"default", _params} ->
        nil

      {"linear", %{"factor" => factor}} ->
        %{type: :linear, factor: validate_factor!(factor, "linear")}

      {"dynamic", %{"factor" => factor}} ->
        %{type: :dynamic, factor: validate_factor!(factor, type)}

      {"llama3", %{"factor" => factor, "low_freq_factor" => low, "high_freq_factor" => high}} ->
        llama3_scaling_strategy(factor, low, high, original_max_positions)

      # Old Phi-3 checkpoints use "su" and "yarn" for LongRoPE
      {type, %{"short_factor" => _short_factor, "long_factor" => _long_factor} = params}
      when type in ["longrope", "su", "yarn"] ->
        longrope_scaling_strategy(params, original_max_positions)

      {"yarn", %{"factor" => _factor}} = config ->
        yarn_scaling_strategy(config, original_max_positions)

      _other ->
        raise "conversion failed, unsupported rotary embedding parameters: #{inspect(params)}"
    end
  end

  defp yarn_scaling_strategy({"yarn", params}, original_max_positions) do
    factor = validate_factor!(params["factor"], "yarn")
    original_max_positions = validate_original_max_positions!(original_max_positions, "yarn")
    beta_fast = validate_positive_number!(params["beta_fast"] || 32.0, "beta_fast", "yarn")
    beta_slow = validate_positive_number!(params["beta_slow"] || 1.0, "beta_slow", "yarn")

    unless is_boolean(Map.get(params, "truncate", true)) do
      raise "conversion failed, yarn requires \"truncate\" to be a boolean"
    end

    if beta_fast < beta_slow do
      raise "conversion failed, YaRN requires \"beta_fast\" >= \"beta_slow\""
    end

    %{
      type: :yarn,
      factor: factor,
      original_max_positions: original_max_positions,
      beta_fast: beta_fast,
      beta_slow: beta_slow,
      attention_factor: yarn_attention_factor(params, factor),
      truncate: Map.get(params, "truncate", true)
    }
  end

  defp llama3_scaling_strategy(factor, low, high, original_max_positions) do
    factor = validate_factor!(factor, "llama3")
    low = validate_positive_number!(low, "low_freq_factor", "llama3")
    high = validate_positive_number!(high, "high_freq_factor", "llama3")

    if high <= low do
      raise "conversion failed, llama3 requires \"high_freq_factor\" > \"low_freq_factor\""
    end

    %{
      type: :llama3,
      factor: factor,
      low_frequency_factor: low,
      high_frequency_factor: high,
      original_max_positions: validate_original_max_positions!(original_max_positions, "llama3")
    }
  end

  defp rope_type!(%{"rope_type" => rope_type, "type" => type}) when rope_type != type do
    raise "conversion failed, conflicting \"rope_type\" and \"type\" values"
  end

  defp rope_type!(%{"rope_type" => rope_type}), do: rope_type
  defp rope_type!(%{"type" => type}), do: type
  defp rope_type!(_params), do: "default"

  defp scaling_fields?(params) do
    Enum.any?(
      [
        "factor",
        "short_factor",
        "long_factor",
        "low_freq_factor",
        "high_freq_factor",
        "beta_fast",
        "beta_slow",
        "attention_factor",
        "mscale",
        "mscale_all_dim",
        "truncate"
      ],
      &Map.has_key?(params, &1)
    )
  end

  defp longrope_scaling_strategy(params, original_max_positions) do
    %{
      type: :longrope,
      short_factor: validate_number_list!(params["short_factor"], "short_factor", "longrope"),
      long_factor: validate_number_list!(params["long_factor"], "long_factor", "longrope"),
      original_max_positions: validate_original_max_positions!(original_max_positions, "longrope")
    }
  end

  defp yarn_attention_factor(params, factor) do
    case params["attention_factor"] do
      nil ->
        case {params["mscale"], params["mscale_all_dim"]} do
          {nil, nil} ->
            yarn_mscale(factor)

          {mscale, mscale_all_dim} when is_number(mscale) and is_number(mscale_all_dim) ->
            yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim)

          _other ->
            raise "conversion failed, yarn requires numeric \"mscale\" and \"mscale_all_dim\" together"
        end

      attention_factor ->
        validate_positive_number!(attention_factor, "attention_factor", "yarn")
    end
  end

  defp yarn_mscale(factor, mscale \\ 1.0)
  defp yarn_mscale(factor, _mscale) when factor <= 1.0, do: 1.0
  defp yarn_mscale(factor, mscale), do: 0.1 * mscale * :math.log(factor) + 1.0

  defp validate_factor!(factor, type) do
    if is_number(factor) and factor >= 1.0 do
      factor
    else
      raise "conversion failed, #{type} requires a numeric \"factor\" >= 1"
    end
  end

  defp validate_original_max_positions!(value, type) do
    if is_integer(value) and value > 0 do
      value
    else
      raise "conversion failed, #{type} requires a positive integer \"original_max_position_embeddings\""
    end
  end

  defp validate_positive_number!(value, key, type) do
    if is_number(value) and value > 0 do
      value
    else
      raise "conversion failed, #{type} requires a positive numeric #{inspect(key)}"
    end
  end

  defp validate_number_list!(value, key, type) do
    if is_list(value) and value != [] and Enum.all?(value, &(is_number(&1) and &1 > 0)) do
      value
    else
      raise "conversion failed, #{type} requires a non-empty list of positive numbers for #{inspect(key)}"
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

  def validate_string_or_pairs(input) do
    case input do
      input when is_binary(input) -> {:ok, input}
      {left, right} when is_binary(left) and is_binary(right) -> {:ok, input}
      _other -> {:error, "expected a string or a pair of strings, got: #{inspect(input)}"}
    end
  end

  @doc """
  Validates that the input is a single value and not a batch.
  """
  @spec validate_input_for_stream!(term()) :: :ok
  def validate_input_for_stream!(input) do
    if is_list(input) do
      raise ArgumentError,
            "this serving only accepts singular input when stream is enabled," <>
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

  If `defn_options[:cache]` is set, the given `scope` is used to create
  a suffix.
  """
  @spec compile_or_jit(
          function(),
          scope,
          keyword(),
          boolean(),
          (-> list(Nx.Tensor.t()))
        ) :: function()
        when scope: String.Chars.t() | {scope, scope}
  def compile_or_jit(fun, scope, defn_options, compile?, template_fun) do
    defn_options =
      case defn_options[:cache] do
        cache when is_binary(cache) ->
          suffix = "__bumblebee_" <> scope_to_string(scope)
          Keyword.replace!(defn_options, :cache, cache <> suffix)

        _ ->
          defn_options
      end

    if compile? do
      template_args = template_fun.() |> templates()
      Nx.Defn.compile(fun, template_args, defn_options)
    else
      Nx.Defn.jit(fun, defn_options)
    end
  end

  defp scope_to_string({left, right}) do
    scope_to_string(left) <> "_" <> scope_to_string(right)
  end

  defp scope_to_string(scope), do: to_string(scope)

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
    {height, width} = images_spatial_sizes(images)

    {short, long} = if height < width, do: {height, width}, else: {width, height}

    out_short = size
    out_long = floor(size * long / short)

    if height < width, do: {out_short, out_long}, else: {out_long, out_short}
  end

  defp images_spatial_sizes(images) do
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
