defmodule Bumblebee.Text.GlinerEntityExtraction do
  @moduledoc """
  GLiNER 2.5 entity extraction with labels supplied for each document.

  See `Bumblebee.Text.load_entity_extraction/2` and
  `Bumblebee.Text.entity_extraction/2` for loading and serving.
  """
  @behaviour Nx.Serving
  alias Bumblebee.Text.{DebertaV2, Gliner}
  alias Gliner.{CandidatePool, Decode, Preprocessor}

  defstruct [
    :encoder,
    :boundary,
    :scorer,
    :tokenizer,
    :config,
    :defn_options,
    :sequence_length,
    :max_sequence_length
  ]

  @doc false
  def load(repository, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :backend,
        :log_params_diff,
        defn_options: [],
        sequence_length: nil,
        max_sequence_length: 4096
      ])

    sequence_length = opts[:sequence_length]
    maximum = opts[:max_sequence_length]

    unless is_integer(maximum) and maximum > 0,
      do: raise(ArgumentError, "max_sequence_length must be a positive integer")

    unless is_nil(sequence_length) or
             (is_integer(sequence_length) and sequence_length > 0 and sequence_length <= maximum),
           do:
             raise(
               ArgumentError,
               "sequence_length must be positive and at most max_sequence_length"
             )

    with {:ok, encoder_spec} <-
           Bumblebee.load_spec(encoder_repository(repository),
             module: DebertaV2,
             architecture: :base
           ),
         {:ok, spec} <- Bumblebee.load_spec(repository, module: Gliner, architecture: :boundary),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(repository, type: :deberta_v2) do
      validate_tokenizer!(tokenizer.native_tokenizer)

      spec =
        Bumblebee.configure(spec,
          hidden_size: encoder_spec.hidden_size,
          encoder_spec: encoder_spec
        )

      load_opts = Keyword.take(opts, [:backend, :log_params_diff])

      with {:ok, encoder} <-
             load_stage(repository, spec, :encoder, load_opts, opts[:defn_options]),
           {:ok, boundary} <-
             load_stage(repository, spec, :boundary, load_opts, opts[:defn_options]),
           {:ok, scorer} <-
             load_stage(repository, spec, :span_scoring, load_opts, opts[:defn_options]) do
        {:ok,
         %__MODULE__{
           encoder: encoder,
           boundary: boundary,
           scorer: scorer,
           tokenizer: tokenizer.native_tokenizer,
           config: Gliner.boundary_config(spec),
           defn_options: opts[:defn_options],
           sequence_length: sequence_length,
           max_sequence_length: maximum
         }}
      end
    end
  end

  defp validate_tokenizer!(tokenizer) do
    for token <- ["[P]", "[E]", "[SEP_TEXT]"] do
      if Tokenizers.Tokenizer.token_to_id(tokenizer, token) == nil do
        raise ArgumentError, "GLiNER tokenizer is missing #{token}"
      end
    end
  end

  defp load_stage(repository, spec, architecture, load_opts, defn_options) do
    spec = Bumblebee.configure(spec, architecture: architecture)

    with {:ok, info} <- Bumblebee.load_model(repository, [spec: spec] ++ load_opts) do
      {_, predict} = Axon.build(info.model, defn_options)
      {:ok, {predict, info.params}}
    end
  end

  defp encoder_repository({:local, path}), do: {:local, Path.join(path, "encoder_config")}
  defp encoder_repository({:hf, id}), do: encoder_repository({:hf, id, []})

  defp encoder_repository({:hf, id, opts}) do
    subdir = if opts[:subdir], do: opts[:subdir] <> "/encoder_config", else: "encoder_config"
    {:hf, id, Keyword.put(opts, :subdir, subdir)}
  end

  @doc """
  Extracts entities from one document without starting a serving process.

  Options match `Bumblebee.Text.entity_extraction/2`. Offsets count Unicode
  codepoints in the original text. Labels are deduplicated in input order.
  """
  def extract_entities(model, text, labels, opts \\ []) do
    labels = validate_input!(%{text: text, labels: labels})
    opts = validate_options!(opts)

    if labels == [] do
      %{}
    else
      packed = pack(model, text, labels)
      Decode.entities(packed, infer(model, packed), model.config, opts)
    end
  end

  @doc false
  def new(model, opts \\ []) do
    opts = validate_options!(opts)

    Nx.Serving.new(__MODULE__, model)
    |> Nx.Serving.batch_size(1)
    |> Nx.Serving.process_options(batch_keys: [:default, :empty])
    |> Nx.Serving.client_preprocessing(fn input ->
      labels = validate_input!(input)

      if labels == [] do
        {Nx.Batch.concatenate([Nx.tensor([0])]) |> Nx.Batch.key(:empty), :empty}
      else
        packed = pack(model, input.text, labels)
        data = {packed.inputs, packed.text_indices, packed.query_indices}
        {Nx.Batch.concatenate([data]), packed}
      end
    end)
    |> Nx.Serving.client_postprocessing(fn
      {_outputs, _metadata}, :empty -> %{}
      {outputs, _metadata}, packed -> Decode.entities(packed, outputs, model.config, opts)
    end)
  end

  @impl true
  def init(_type, model, _partitions), do: {:ok, model}

  @impl true
  def handle_batch(%Nx.Batch{key: :empty}, _partition, model),
    do: {:execute, fn -> {Nx.tensor([0]), :inference} end, model}

  def handle_batch(%Nx.Batch{size: 1} = batch, _partition, model) do
    execute = fn ->
      {inputs, text_indices, query_indices} =
        Nx.Defn.jit_apply(&Function.identity/1, [batch], model.defn_options)

      outputs =
        infer(model, %{inputs: inputs, text_indices: text_indices, query_indices: query_indices})

      {Map.take(outputs, ~w(pair_logits null_logits starts ends valid)), :inference}
    end

    {:execute, execute, model}
  end

  def handle_batch(_batch, _partition, _model),
    do: raise(ArgumentError, "GLiNER serving requires batch_size: 1")

  @doc false
  def infer(model, packed) do
    original_length = Nx.axis_size(packed.inputs["input_ids"], 1)
    hidden = predict(model.encoder, pad_inputs(packed.inputs, model.sequence_length)).hidden_state
    hidden = Nx.slice_along_axis(hidden, 0, original_length, axis: 1)

    inputs = %{
      "text_states" => route(hidden, packed.text_indices),
      "query_states" => route(hidden, packed.query_indices)
    }

    output = predict(model.boundary, inputs)
    candidates = CandidatePool.select(output, model.config)
    inputs = inputs |> Map.merge(output) |> Map.merge(candidates)

    inputs
    |> Map.put("pair_logits", predict(model.scorer, inputs))
    |> Map.put("hidden_states", hidden)
  end

  defp predict({fun, params}, inputs), do: fun.(params, inputs)
  defp route(hidden, indices), do: Nx.take(hidden, Nx.squeeze(indices, axes: [0]), axis: 1)

  defp pack(model, text, labels) do
    packed = Preprocessor.pack(text, labels, model.tokenizer)
    limit = model.sequence_length || model.max_sequence_length

    if Nx.axis_size(packed.inputs["input_ids"], 1) > limit do
      raise ArgumentError, "packed input exceeds the configured limit of #{limit} tokens"
    end

    packed
  end

  defp pad_inputs(inputs, nil), do: inputs

  defp pad_inputs(inputs, length) do
    padding = length - Nx.axis_size(inputs["input_ids"], 1)
    if padding < 0, do: raise(ArgumentError, "input exceeds sequence_length")

    Map.new(inputs, fn {key, tensor} -> {key, Nx.pad(tensor, 0, [{0, 0, 0}, {0, padding, 0}])} end)
  end

  defp validate_input!(%{text: text, labels: labels}) when is_binary(text) and is_list(labels) do
    unless Enum.all?(labels, &(is_binary(&1) and String.trim(&1) != "")),
      do: raise(ArgumentError, "labels must be nonempty strings")

    Enum.uniq(labels)
  end

  defp validate_input!(_),
    do: raise(ArgumentError, "expected %{text: string, labels: list_of_strings}")

  defp validate_options!(opts) do
    opts =
      Keyword.validate!(opts, threshold: 0.5, include_spans: false, include_confidence: false)

    unless is_number(opts[:threshold]) and opts[:threshold] >= 0 and opts[:threshold] <= 1,
      do: raise(ArgumentError, "threshold must be between zero and one")

    for key <- [:include_spans, :include_confidence],
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{key} must be a boolean")

    opts
  end
end
