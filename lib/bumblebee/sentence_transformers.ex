defmodule Bumblebee.SentenceTransformers do
  @moduledoc """
  SentenceTransformers pipelines in Elixir on top of Bumblebee, Axon, and Nx.

  SentenceTransformers supports a subset of Python's `sentence-transformers` inference modules
  by parsing `modules.json` and downloading component files on demand
  (e.g. `1_Pooling/config.json`, `2_Dense/model.safetensors`).

  ## Quick start

      {:ok, model_info} = Bumblebee.SentenceTransformers.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})
      {:ok, tokenizer} = Bumblebee.SentenceTransformers.load_tokenizer({:hf, "sentence-transformers/all-MiniLM-L6-v2"})

      serving = Bumblebee.SentenceTransformers.text_embedding(model_info, tokenizer)
      Nx.Serving.run(serving, "Hello world")
      #=> %{embedding: #Nx.Tensor<f32[384] ...>}

  `text_embedding/3` uses a saved default prompt and saved maximum sequence
  length when the caller does not override them.
  """

  alias Bumblebee.SentenceTransformers.Downloader
  alias Bumblebee.SentenceTransformers.Pipeline
  alias Bumblebee.Text.PreTrainedTokenizer

  @type repository :: Bumblebee.repository()

  @doc """
  Loads a SentenceTransformers model from a repository.

  If `modules.json` is present in the repository, SentenceTransformers parses the module pipeline,
  loads the underlying base model, and stacks the declared modules (pooling, dense
  projections, normalization, etc.) on top.

  If `modules.json` is not present, it applies fallback pooling: `:last` for
  causal language models and `:mean` for other model families.

  ## Options

    * `:fuse_dense` - whether to pre-multiply consecutive linear projection layers
      without bias into a single projection layer. Defaults to `false`

    * `:type` - the numerical precision or Axon mixed precision policy

    * `:backend` - the Nx backend to allocate parameters on

    * `:safetensors_reader` - function used to read `.safetensors` files

  """
  @spec load_model(repository(), keyword()) :: {:ok, map()} | {:error, String.t()}
  def load_model(repository, opts \\ []) do
    base_opts =
      opts
      |> Keyword.take([:backend, :type, :safetensors_reader])
      |> Keyword.put_new(:architecture, :base)

    case load_pipeline_modules(repository) do
      {:ok, [first | pipeline_modules]} ->
        with {:ok, model_info} <- load_base_model(repository, first, base_opts, opts),
             {:ok, model_info} <- Pipeline.build(repository, pipeline_modules, model_info, opts) do
          load_config_and_attach(repository, model_info)
        end

      {:missing_modules, _reason} ->
        load_fallback_model(repository, base_opts, opts)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp load_base_model(repository, %{"path" => path, "type" => type} = first, base_opts, opts) do
    case Pipeline.normalize_module_type(type) do
      "Transformer" ->
        base_repository = Downloader.subrepository(repository, path)
        Bumblebee.load_model(base_repository, base_opts)

      "StaticEmbedding" ->
        Pipeline.build_static_embedding(repository, first, opts)

      other ->
        {:error, "unsupported base module in modules.json: #{inspect(other)}"}
    end
  end

  defp load_base_model(_repository, _first, _base_opts, _opts) do
    {:error, "expected modules.json to define a Transformer or StaticEmbedding base model"}
  end

  defp load_fallback_model(repository, base_opts, opts) do
    # Fallback when modules.json is not present: load model and apply default pooling
    # CausalLM-based models use last token pooling, otherwise mean pooling
    with {:ok, model_info} <- Bumblebee.load_model(repository, base_opts),
         fallback_modules = build_fallback_modules(repository, model_info),
         {:ok, model_info} <- Pipeline.build(repository, fallback_modules, model_info, opts) do
      load_config_and_attach(repository, model_info)
    end
  end

  defp build_fallback_modules(repository, model_info) do
    pooling_mode = infer_fallback_pooling_mode(repository, model_info)
    [%{"type" => "Pooling", "path" => "default_pooling", "pooling_mode" => pooling_mode}]
  end

  defp load_pipeline_modules(repository, opts \\ []) do
    case Downloader.download_file(repository, "modules.json", opts) do
      {:ok, path} ->
        with {:ok, modules} <- Pipeline.decode_json(path) do
          case Enum.sort_by(modules, &(&1["idx"] || 0)) do
            [] -> {:error, "modules.json is empty"}
            sorted -> {:ok, sorted}
          end
        end

      {:error, reason} ->
        if Downloader.missing_file?(repository, "modules.json", reason) do
          {:missing_modules, reason}
        else
          {:error, reason}
        end
    end
  end

  @doc """
  Loads only the SentenceTransformers embedding head and attaches it to an existing `model_info`.

  Supports piping directly from `Bumblebee.load_model/2`:

      {:ok, model_info} =
        Bumblebee.load_model(repo, architecture: :base)
        |> Bumblebee.SentenceTransformers.load_embedding_head(repo)

  """
  def load_embedding_head(model_or_repo, repo_or_model, opts \\ [])

  def load_embedding_head({:ok, model_info}, repository, opts) when is_map(model_info) do
    load_embedding_head(model_info, repository, opts)
  end

  def load_embedding_head({:error, reason}, _repository, _opts) do
    {:error, reason}
  end

  def load_embedding_head(%{model: _} = model_info, repository, opts) do
    do_load_embedding_head(repository, model_info, opts)
  end

  def load_embedding_head(repository, %{model: _} = model_info, opts) do
    do_load_embedding_head(repository, model_info, opts)
  end

  def load_embedding_head(repository, {:ok, %{model: _} = model_info}, opts) do
    do_load_embedding_head(repository, model_info, opts)
  end

  @doc """
  Alias for `load_embedding_head/3`.
  """
  def load_head(model_or_repo, repo_or_model, opts \\ []) do
    load_embedding_head(model_or_repo, repo_or_model, opts)
  end

  defp do_load_embedding_head(repository, model_info, opts) do
    case load_pipeline_modules(repository) do
      {:ok, [first | pipeline_modules]} ->
        if Pipeline.normalize_module_type(first["type"]) in ["Transformer", "StaticEmbedding"] do
          with {:ok, model_info} <- Pipeline.build(repository, pipeline_modules, model_info, opts) do
            load_config_and_attach(repository, model_info)
          end
        else
          {:error,
           "expected the first module in modules.json to be Transformer or StaticEmbedding"}
        end

      {:missing_modules, _reason} ->
        {:error, "could not find modules.json in the repository"}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Loads the tokenizer for the first module in a SentenceTransformers pipeline,
  or the repository root when `modules.json` is absent.
  """
  def load_tokenizer(repository, opts \\ []) do
    case load_pipeline_modules(repository, opts) do
      {:ok, [first | _]} ->
        Bumblebee.load_tokenizer(Downloader.subrepository(repository, first["path"]), opts)

      {:missing_modules, _reason} ->
        Bumblebee.load_tokenizer(repository, opts)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Builds a text embedding serving that returns `%{embedding: tensor}` for one
  input and a list of those maps for a list of inputs. Set
  `return_tokenized: true` to also return each input's tokenized
  `%{"input_ids" => tensor, "attention_mask" => tensor}`.

  This mirrors `Bumblebee.Text.text_embedding/3` and additionally supports
  SentenceTransformers models configured with `include_prompt: false`. In those
  cases the prompt prefix is tokenized separately and a `prompt_length` input is
  passed to the model so that prompt tokens are excluded from pooling.

  ## Options

    * `:prompt` - a custom prompt string to prepend to every input. It takes
      precedence over the saved default prompt. Pass `false` or `nil` to
      disable prompt use.

    * `:max_seq_length` - overrides the saved token limit. Compilation bucket
      sizes control padding; they do not increase this limit. Smaller buckets
      cap it at the largest available bucket.

    * `:prompt_name` - a saved prompt name from
      `config_sentence_transformers.json`. It cannot be combined with
      `:prompt`; pass `false` to disable prompt use. With neither
      option, the saved default prompt is used.

    * `:output_attribute` - model output key to embed. Defaults to
      `:pooled_state`.

    * `:output_pool` - optional `:cls_token_pooling`, `:mean_pooling`, or
      `:last_token_pooling` for rank-3 outputs. Defaults to `nil`.

    * `:embedding_processor` - optional `:l2_norm`. Defaults to `nil`.

    * `:return_tokenized` - includes tokenized inputs in each result map.
      Defaults to `false`.

    * `:compile`, `:defn_options`, `:preallocate_params` - serving execution
      options. A compile sequence length sets a padding bucket and caps the
      effective saved or requested maximum sequence length.

  """
  def text_embedding(model_info, tokenizer, opts \\ []) do
    tokenizer =
      if model_info[:sparse_static_embedding] do
        Bumblebee.configure(tokenizer, add_special_tokens: false)
      else
        tokenizer
      end

    text_embedding_with_prompt_length(model_info, tokenizer, opts)
  end

  @doc false
  def configure_embedding_tokenizer(model_info, tokenizer, opts) do
    saved = get_in(model_info, [:sentence_transformers, :max_seq_length])
    limit = Keyword.get(opts, :max_seq_length, saved)

    if limit != nil and (not is_integer(limit) or limit <= 0) do
      raise ArgumentError, ":max_seq_length must be a positive integer"
    end

    buckets = get_in(opts, [:compile, :sequence_length])
    upper = if buckets, do: Enum.max(List.wrap(buckets))
    limit = if limit && upper, do: min(limit, upper), else: limit || upper
    tokenizer = Bumblebee.configure(tokenizer, length: nil, return_token_type_ids: false)

    if limit do
      # Bumblebee's length option couples truncation with fixed padding. Set
      # native truncation alone so short inputs still use dynamic/small buckets.
      update_in(tokenizer.native_tokenizer, fn native ->
        Tokenizers.Tokenizer.set_truncation(native,
          max_length: limit,
          direction: tokenizer.truncate_direction
        )
      end)
    else
      tokenizer
    end
  end

  @doc false
  def apply_embedding_tokenizer(tokenizer, texts, buckets) do
    inputs = Bumblebee.apply_tokenizer(tokenizer, texts)

    if buckets do
      length = Nx.axis_size(inputs["input_ids"], 1)
      target = buckets |> List.wrap() |> Enum.sort() |> Enum.find(&(&1 >= length))

      pad_id =
        PreTrainedTokenizer.token_to_id(
          tokenizer,
          tokenizer.special_tokens[:pad] || tokenizer.special_tokens[:eos]
        )

      Map.new(inputs, fn {key, tensor} ->
        value = if key == "input_ids", do: pad_id, else: 0

        padding =
          if tokenizer.pad_direction == :left,
            do: {target - length, 0, 0},
            else: {0, target - length, 0}

        {key, Nx.pad(tensor, value, [{0, 0, 0}, padding])}
      end)
    else
      inputs
    end
  end

  @doc """
  Builds a cross encoding serving using `Bumblebee.Text.cross_encoding/3`.
  """
  defdelegate cross_encoding(model_info, tokenizer, opts \\ []), to: Bumblebee.Text

  @doc """
  Encodes query text using the saved `"query"` prompt when available.

  This is a convenience wrapper around `text_embedding/3`. If no `"query"`
  prompt is configured, it falls back to the saved default prompt (if any).

  Explicit `:prompt` or `:prompt_name` options take precedence. Returns the
  same `%{embedding: tensor}` result shape as `text_embedding/3`.
  """
  def encode_queries(model_info, tokenizer, queries, opts \\ []) do
    serving =
      text_embedding(model_info, tokenizer, encoding_prompt_opts(model_info, opts, ["query"]))

    Nx.Serving.run(serving, queries)
  end

  @doc """
  Encodes corpus/document text using the first available saved prompt named
  `"document"`, `"passage"`, or `"corpus"`.

  If none of those aliases are configured, it falls back to the saved default
  prompt (if any).

  Explicit `:prompt` or `:prompt_name` options take precedence. Returns the
  same `%{embedding: tensor}` result shape as `text_embedding/3`.
  """
  def encode_corpus(model_info, tokenizer, corpus, opts \\ []) do
    serving =
      text_embedding(
        model_info,
        tokenizer,
        encoding_prompt_opts(model_info, opts, ["document", "passage", "corpus"])
      )

    Nx.Serving.run(serving, corpus)
  end

  defp encoding_prompt_opts(model_info, opts, names) do
    prompts = get_in(model_info, [:sentence_transformers, :prompts]) || %{}

    if Keyword.has_key?(opts, :prompt) or Keyword.has_key?(opts, :prompt_name) do
      opts
    else
      case Enum.find(names, &Map.has_key?(prompts, &1)) do
        nil -> opts
        name -> Keyword.put(opts, :prompt_name, name)
      end
    end
  end

  defp text_embedding_with_prompt_length(model_info, tokenizer, opts) do
    %{model: model, params: params, spec: _spec} = model_info

    opts =
      Keyword.validate!(opts, [
        :compile,
        :prompt,
        :prompt_name,
        :max_seq_length,
        return_tokenized: false,
        output_attribute: :pooled_state,
        output_pool: nil,
        embedding_processor: nil,
        defn_options: [],
        preallocate_params: false
      ])

    prompt_prefix = resolve_prompt(opts, model_info)

    output_attribute = opts[:output_attribute]
    output_pool = opts[:output_pool]
    embedding_processor = opts[:embedding_processor]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size, :sequence_length])
        |> Bumblebee.Shared.require_options!([:batch_size, :sequence_length])
      end

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    tokenizer =
      configure_embedding_tokenizer(model_info, tokenizer, opts)

    prompt_length = prompt_token_length(tokenizer, prompt_prefix)

    {_init_fun, encoder} = Axon.build(model)

    embedding_fun = fn params, inputs ->
      encoder.(params, inputs)
      |> extract_output_tensor(output_attribute)
      |> apply_output_pool(output_pool, inputs)
      |> apply_embedding_processor(embedding_processor)
    end

    batch_keys = Bumblebee.Shared.sequence_batch_keys(sequence_length)

    fn batch_key, defn_options ->
      params = Bumblebee.Shared.maybe_preallocate(params, preallocate_params, defn_options)

      scope = {:embedding, batch_key}

      embedding_fun =
        Bumblebee.Shared.compile_or_jit(
          embedding_fun,
          scope,
          defn_options,
          compile != nil,
          fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32),
              "prompt_length" => Nx.template({batch_size}, :u32)
            }

            [params, inputs]
          end
        )

      fn inputs ->
        inputs = Bumblebee.Shared.maybe_pad(inputs, batch_size)
        params |> embedding_fun.(inputs) |> Bumblebee.Shared.serving_post_computation()
      end
    end
    |> Nx.Serving.new(defn_options)
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} =
        Bumblebee.Shared.validate_serving_input!(input, &Bumblebee.Shared.validate_string/1)

      texts =
        case prompt_prefix do
          "" -> texts
          nil -> texts
          prefix -> Enum.map(texts, &(prefix <> &1))
        end

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          apply_embedding_tokenizer(tokenizer, texts, sequence_length)
        end)

      batch_size = Nx.axis_size(inputs["input_ids"], 0)
      inputs = Map.put(inputs, "prompt_length", Nx.broadcast(prompt_length, {batch_size}))

      batch_key = Bumblebee.Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      tokenized =
        if opts[:return_tokenized], do: Map.take(inputs, ["input_ids", "attention_mask"])

      {batch, {multi?, tokenized}}
    end)
    |> Nx.Serving.client_postprocessing(fn {embeddings, _metadata}, {multi?, tokenized} ->
      embeddings
      |> Bumblebee.Utils.Nx.batch_to_list()
      |> Enum.with_index()
      |> Enum.map(fn {embedding, index} ->
        if tokenized do
          row =
            Map.new(tokenized, fn {key, tensor} ->
              {key, Nx.slice_along_axis(tensor, index, 1, axis: 0)}
            end)

          %{embedding: embedding, tokenized: row}
        else
          %{embedding: embedding}
        end
      end)
      |> Bumblebee.Shared.normalize_output(multi?)
    end)
  end

  defp prompt_token_length(_tokenizer, ""), do: Nx.tensor(0, type: :u32)
  defp prompt_token_length(_tokenizer, nil), do: Nx.tensor(0, type: :u32)

  defp prompt_token_length(tokenizer, prompt) do
    tokenizer = Bumblebee.configure(tokenizer, length: nil)
    inputs = Bumblebee.apply_tokenizer(tokenizer, [prompt])
    length = Nx.axis_size(inputs["input_ids"], 1)

    last_token =
      inputs["input_ids"]
      |> Nx.slice_along_axis(length - 1, 1, axis: 1)
      |> Nx.squeeze()
      |> Nx.to_number()

    special_ids = special_token_ids(tokenizer)

    length =
      if last_token in special_ids do
        length - 1
      else
        length
      end

    Nx.as_type(length, :u32)
  end

  defp special_token_ids(tokenizer) do
    tokenizer.special_tokens
    |> Map.values()
    |> Enum.concat(tokenizer.additional_special_tokens)
    |> MapSet.new(&PreTrainedTokenizer.token_to_id(tokenizer, &1))
  end

  defp extract_output_tensor(%Nx.Tensor{} = tensor, _attribute), do: tensor

  defp extract_output_tensor(%{} = output, attribute) do
    case Map.fetch(output, attribute) do
      {:ok, %Axon.None{}} ->
        keys = output |> Map.keys() |> Enum.sort()

        raise ArgumentError,
              "key #{inspect(attribute)} in the output map has value %Axon.None{}," <>
                " make sure this is the correct key and check module documentation incase it is opt-in." <>
                " The output map keys are: #{inspect(keys)}"

      {:ok, tensor} ->
        tensor

      :error when attribute == :pooled_state and is_map_key(output, :embedding) ->
        output.embedding

      :error ->
        keys = output |> Map.keys() |> Enum.sort()

        raise ArgumentError,
              "key #{inspect(attribute)} not found in the output map," <>
                " you may want to set :output_attribute to one of the map keys: #{inspect(keys)}"
    end
  end

  defp extract_output_tensor(other, _attribute), do: other

  defp apply_output_pool(output, nil, _inputs), do: output

  defp apply_output_pool(output, pool, inputs) do
    if Nx.rank(output) != 3 do
      raise ArgumentError,
            "expected the output tensor to have rank 3 to apply :output_pool, got: #{Nx.rank(output)}." <>
              " You should either disable pooling or pick a different output using :output_attribute"
    end

    case pool do
      :cls_token_pooling ->
        Nx.take(output, 0, axis: 1)

      :mean_pooling ->
        input_mask_expanded = Nx.new_axis(inputs["attention_mask"], -1)

        output
        |> Nx.multiply(input_mask_expanded)
        |> Nx.sum(axes: [1])
        |> Nx.divide(Nx.sum(input_mask_expanded, axes: [1]))

      :last_token_pooling ->
        sequence_lengths =
          inputs["attention_mask"]
          |> Nx.sum(axes: [1])
          |> Nx.subtract(1)
          |> Nx.as_type({:s, 64})

        Bumblebee.Utils.Nx.batched_take(output, sequence_lengths)

      other ->
        raise ArgumentError,
              "expected :output_pool to be one of :cls_token_pooling, :mean_pooling, :last_token_pooling or nil, got: #{inspect(other)}"
    end
  end

  defp apply_embedding_processor(output, nil), do: output
  defp apply_embedding_processor(output, :l2_norm), do: Bumblebee.Utils.Nx.normalize(output)

  defp apply_embedding_processor(_output, other) do
    raise ArgumentError,
          "expected :embedding_processor to be one of nil or :l2_norm, got: #{inspect(other)}"
  end

  defp resolve_prompt(opts, model_info) do
    case {Keyword.fetch(opts, :prompt), Keyword.fetch(opts, :prompt_name)} do
      {{:ok, prompt}, :error} when is_binary(prompt) ->
        prompt

      {{:ok, prompt}, :error} when prompt in [false, nil] ->
        ""

      {{:ok, prompt}, :error} ->
        raise ArgumentError, "expected :prompt to be a string or false, got: #{inspect(prompt)}"

      {:error, {:ok, prompt_name}} when is_binary(prompt_name) ->
        find_prompt_by_name!(model_info, prompt_name)

      {:error, {:ok, prompt_name}} when prompt_name in [false, nil] ->
        ""

      {:error, {:ok, prompt_name}} ->
        raise ArgumentError,
              "expected :prompt_name to be a string or false, got: #{inspect(prompt_name)}"

      {:error, :error} ->
        find_default_prompt(model_info)

      {{:ok, _}, {:ok, _}} ->
        raise ArgumentError, "expected either :prompt or :prompt_name, but both were given"
    end
  end

  defp find_prompt_by_name!(model_info, prompt_name) do
    prompts = get_in(model_info, [:sentence_transformers, :prompts]) || %{}

    case Map.fetch(prompts, prompt_name) do
      {:ok, template} ->
        template

      :error ->
        raise ArgumentError,
              "unknown prompt name #{inspect(prompt_name)}. " <>
                "Available prompts: #{inspect(Map.keys(prompts))}"
    end
  end

  defp find_default_prompt(model_info) do
    st = Map.get(model_info, :sentence_transformers, %{}) || %{}

    if default_name = st[:default_prompt_name] do
      get_in(st, [:prompts, default_name])
    end
  end

  @doc """
  Truncates embeddings along the last dimension to `truncate_dim` (Matryoshka representation).

  ## Options

    * `:normalize` - whether to re-apply L2 normalization after truncation.
      Defaults to `false`.

  """
  def truncate_embeddings(embeddings, truncate_dim, opts \\ [])
      when is_integer(truncate_dim) and truncate_dim > 0 do
    normalize? = Keyword.get(opts, :normalize, false)
    truncated = Nx.slice_along_axis(embeddings, 0, truncate_dim, axis: -1)

    if normalize? do
      Bumblebee.Utils.Nx.normalize(truncated)
    else
      truncated
    end
  end

  @doc """
  Loads root prompt metadata and the first module's saved token limit independently.
  """
  def load_config(repository, opts \\ []) do
    with {:ok, root} <- optional_config(repository, "config_sentence_transformers.json", opts),
         {:ok, legacy} <- optional_config(repository, "sentence_bert_config.json", opts),
         {:ok, module_path} <- first_module_path(repository, opts),
         {:ok, transformer} <-
           optional_config(repository, Path.join(module_path, "sentence_bert_config.json"), opts),
         {:ok, tokenizer} <-
           optional_config(repository, Path.join(module_path, "tokenizer_config.json"), opts),
         {:ok, root_tokenizer} <- optional_config(repository, "tokenizer_config.json", opts) do
      configs = Enum.reject([transformer, legacy, root], &is_nil/1)

      if configs == [] and tokenizer == nil and root_tokenizer == nil do
        {:error, "could not find sentence transformers config"}
      else
        config = Enum.reduce(configs, %{}, &Map.merge(&2, &1))
        limit = resolve_max_seq_length(transformer, config, tokenizer, root_tokenizer)
        {:ok, parse_st_config(Map.put(config, "max_seq_length", limit))}
      end
    end
  end

  defp resolve_max_seq_length(transformer, config, tokenizer, root_tokenizer) do
    tokenizer_limit = (tokenizer || root_tokenizer || %{})["model_max_length"]

    tokenizer_limit =
      if is_integer(tokenizer_limit) and tokenizer_limit > 0 and tokenizer_limit < 1_000_000_000,
        do: tokenizer_limit

    (transformer || %{})["max_seq_length"] || config["max_seq_length"] || tokenizer_limit
  end

  defp first_module_path(repository, opts) do
    case load_pipeline_modules(repository, opts) do
      {:ok, [first | _]} -> {:ok, first["path"] || ""}
      {:missing_modules, _} -> {:ok, "0_Transformer"}
      {:error, reason} -> {:error, reason}
    end
  end

  defp optional_config(repository, filename, opts) do
    case Downloader.download_file(repository, filename, opts) do
      {:ok, path} ->
        Pipeline.decode_json(path)

      {:error, reason} ->
        if Downloader.missing_file?(repository, filename, reason),
          do: {:ok, nil},
          else: {:error, reason}
    end
  end

  defp load_config_and_attach(repository, model_info) do
    case load_config(repository) do
      {:ok, config} ->
        {:ok, Map.put(model_info, :sentence_transformers, config)}

      {:error, "could not find sentence transformers config"} ->
        {:ok, Map.put(model_info, :sentence_transformers, default_st_config())}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_st_config(config) do
    %{
      prompts: Map.get(config, "prompts", %{}),
      default_prompt_name: config["default_prompt_name"],
      similarity_fn_name: parse_similarity_fn_name(config["similarity_fn_name"]),
      max_seq_length: config["max_seq_length"]
    }
  end

  defp default_st_config do
    %{
      prompts: %{},
      default_prompt_name: nil,
      similarity_fn_name: nil,
      max_seq_length: nil
    }
  end

  defp parse_similarity_fn_name("cosine"), do: :cosine
  defp parse_similarity_fn_name("dot"), do: :dot
  defp parse_similarity_fn_name("euclidean"), do: :euclidean
  defp parse_similarity_fn_name("manhattan"), do: :manhattan
  defp parse_similarity_fn_name(nil), do: nil
  defp parse_similarity_fn_name(other), do: other

  @causal_spec_names [
    "Gemma",
    "GemmaConfig",
    "Gemma3Text",
    "Gemma3TextConfig",
    "Llama",
    "LlamaConfig",
    "Mistral",
    "MistralConfig",
    "Qwen3",
    "Qwen3Config",
    "SmolLm3",
    "Smollm3Config",
    "Phi",
    "PhiConfig",
    "Phi3",
    "Phi3Config",
    "Gpt2",
    "Gpt2Config",
    "GptBigCode",
    "GptBigCodeConfig",
    "GptNeoX",
    "GptNeoXConfig"
  ]

  @doc """
  Infers fallback pooling mode when `modules.json` is missing.
  Returns `"last"` for causal-LM architectures, and `"mean"` otherwise.
  """
  def infer_fallback_pooling_mode(repository, model_info) do
    if causal_from_config?(repository) or causal_from_spec?(model_info) do
      "last"
    else
      "mean"
    end
  end

  defp causal_from_config?(repository) do
    with {:ok, path} <- Downloader.download_file(repository, "config.json"),
         {:ok, config} <- Pipeline.decode_json(path) do
      archs = config["architectures"] || []

      causal_arch? =
        Enum.any?(archs, fn arch ->
          is_binary(arch) and
            (String.ends_with?(arch, "ForCausalLM") or String.ends_with?(arch, "LMHeadModel"))
        end)

      causal_arch? and Map.get(config, "is_causal", true)
    else
      _ -> false
    end
  end

  defp causal_from_spec?(model_info) do
    case model_info[:spec] || model_info[:config] do
      nil ->
        false

      spec ->
        struct_name = spec.__struct__ |> Module.split() |> List.last()
        struct_name in @causal_spec_names
    end
  end
end
