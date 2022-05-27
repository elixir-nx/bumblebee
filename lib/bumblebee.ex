defmodule Bumblebee do
  alias Bumblebee.Utils.HTTP

  @config_filename "config.json"
  @params_filename %{pytorch: "pytorch_model.bin"}

  @huggingface_endpoint "https://huggingface.co"

  # TODO: add support for local files, potentially a pluggable storage

  @doc """
  Loads a pretrained model from a model repository on Hugging Face.

  ## Options

    * `:revision` - the specific model version to use, it can be any
      valid git identifier, such as branch name, tag name, or a commit
      hash

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

  """
  @spec load_model(module(), atom(), String.t(), keyword()) ::
          {:ok, Axon.t(), params :: map(), config :: map()} | {:error, String.t()}
  def load_model(module, model_fun, model_id, opts \\ []) do
    base_model_prefix = module.base_model_prefix()

    with {:ok, config} <- load_config(module, model_id, opts),
         model <- apply(module, model_fun, [[config: config]]),
         {:ok, params} <-
           load_params(model, model_id, [{:base_model_prefix, base_model_prefix} | opts]) do
      {:ok, model, params, config}
    end
  end

  @doc """
  Loads model parameters from a model repository on Hugging Face.

  ## Options

    * `:revision` - the specific model version to use, it can be any
      valid git identifier, such as branch name, tag name, or a commit
      hash

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

    * `:base_model_prefix` - the base model name in layer names.
      Allows for loading base model parameters into specialized model
      and vice versa

  """
  @spec load_params(Axon.t(), String.t(), keyword()) :: {:ok, map()} | {:error, String.t()}
  def load_params(model, model_id, opts \\ []) do
    opts = Keyword.validate!(opts, [:revision, :cache_dir, :base_model_prefix])
    revision = opts[:revision]
    cache_dir = opts[:cache_dir] || default_cache_dir()
    base_model_prefix = opts[:base_model_prefix]
    # TODO: support format: :auto | :axon | :pytorch
    format = :pytorch
    filename = @params_filename[format]

    url = hf_file_url(model_id, filename, revision)

    with {:ok, path} <- cached_download(url, cache_dir) do
      params =
        Bumblebee.Conversion.PyTorch.load_params!(model, path,
          base_model_prefix: base_model_prefix
        )

      {:ok, params}
    end
  end

  @doc """
  Loads model configuration from a model repository on Hugging Face.

  ## Options

    * `:revision` - the specific model version to use, it can be any
      valid git identifier, such as branch name, tag name, or a commit
      hash

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

  """
  @spec load_config(module(), String.t(), keyword()) :: {:ok, map()} | {:error, String.t()}
  def load_config(module, model_id, opts \\ []) do
    opts = Keyword.validate!(opts, [:revision, :cache_dir])
    revision = opts[:revision]
    cache_dir = opts[:cache_dir] || default_cache_dir()

    url = hf_file_url(model_id, @config_filename, revision)

    with {:ok, path} <- cached_download(url, cache_dir) do
      path
      |> File.read!()
      |> Jason.decode()
      |> case do
        {:ok, data} -> {:ok, module.config(data)}
        _ -> {:error, "failed to parse the config file, it is not a valid JSON"}
      end
    end
  end

  defp hf_file_url(model_id, filename, revision) do
    revision = revision || "main"
    @huggingface_endpoint <> "/#{model_id}/resolve/#{revision}/#{filename}"
  end

  defp cached_download(url, cache_dir) do
    File.mkdir_p!(cache_dir)

    with {:ok, etag, download_url} <- head_download(url) do
      metadata_path = Path.join(cache_dir, metadata_filename(url))
      entry_path = Path.join(cache_dir, entry_filename(url, etag))

      case load_json(metadata_path) do
        {:ok, %{"etag" => ^etag}} ->
          {:ok, entry_path}

        :error ->
          tmp_path = get_tmp_path()

          with :ok <- HTTP.download(download_url, tmp_path) |> finish_request() do
            File.rename!(tmp_path, entry_path)
            :ok = store_json(metadata_path, %{"etag" => etag, "url" => url})
            {:ok, entry_path}
          end
      end
    end
  end

  defp head_download(url) do
    with {:ok, response} <- HTTP.request(:head, url, follow_redirects: false) |> finish_request(),
         {:ok, etag} <- fetch_etag(response) do
      download_url =
        if response.status in 300..399 do
          HTTP.get_header(response, "location")
        else
          url
        end

      {:ok, etag, download_url}
    end
  end

  defp finish_request(:ok), do: :ok

  defp finish_request({:ok, response}) when response.status in 100..399, do: {:ok, response}

  defp finish_request({:ok, response}) do
    case HTTP.get_header(response, "x-error-code") do
      "RepoNotFound" -> {:error, "repository not found"}
      "EntryNotFound" -> {:error, "file not found"}
      "RevisionNotFound" -> {:error, "revision not found"}
      _ -> {:error, "HTTP request failed with status #{response.status}"}
    end
  end

  defp finish_request({:error, reason}) do
    {:error, "failed to make an HTTP request, reason: #{inspect(reason)}"}
  end

  defp fetch_etag(response) do
    if etag = HTTP.get_header(response, "x-linked-etag") || HTTP.get_header(response, "etag") do
      {:ok, etag}
    else
      {:error, "no ETag found on the resource"}
    end
  end

  defp metadata_filename(url) do
    encode_url(url) <> ".json"
  end

  defp entry_filename(url, etag) do
    encode_url(url) <> "." <> encode_etag(etag)
  end

  defp encode_url(url) do
    url |> :erlang.md5() |> Base.encode32(case: :lower, padding: false)
  end

  defp encode_etag(etag) do
    Base.encode32(etag, case: :lower, padding: false)
  end

  defp get_tmp_path() do
    random_id = :crypto.strong_rand_bytes(20) |> Base.encode32(case: :lower)
    Path.join(System.tmp_dir!(), "bumblebee_" <> random_id)
  end

  defp load_json(path) do
    case File.read(path) do
      {:ok, content} -> {:ok, Jason.decode!(content)}
      _error -> :error
    end
  end

  defp store_json(path, data) do
    json = Jason.encode!(data)
    File.write(path, json)
  end

  defp default_cache_dir() do
    base_dir = :filename.basedir(:user_cache, "bumblebee")
    Path.join(base_dir, "huggingface")
  end
end
