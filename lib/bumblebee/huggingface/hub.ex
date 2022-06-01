defmodule Bumblebee.HuggingFace.Hub do
  @moduledoc false

  alias Bumblebee.Utils.HTTP

  @huggingface_endpoint "https://huggingface.co"

  @doc """
  Returns a URL pointing to the given file in a Hugging Face repository.
  """
  @spec file_url(String.t(), String.t(), String.t() | nil) :: String.t()
  def file_url(model_id, filename, revision) do
    revision = revision || "main"
    @huggingface_endpoint <> "/#{model_id}/resolve/#{revision}/#{filename}"
  end

  @doc """
  Downloads file from the given URL and returns a path to the file.

  The file is cached based on the received ETag. Subsequent requests
  for the same URL validate the ETag and return a file from the cache
  if there is a match.

  ## Options

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

  """
  @spec cached_download(String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def cached_download(url, opts \\ []) do
    cache_dir = opts[:cache_dir] || default_cache_dir()

    File.mkdir_p!(cache_dir)

    with {:ok, etag, download_url} <- head_download(url) do
      metadata_path = Path.join(cache_dir, metadata_filename(url))
      entry_path = Path.join(cache_dir, entry_filename(url, etag))

      case load_json(metadata_path) do
        {:ok, %{"etag" => ^etag}} ->
          {:ok, entry_path}

        _ ->
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
