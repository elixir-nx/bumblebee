defmodule Bumblebee.HuggingFace.Hub do
  @moduledoc false

  alias Bumblebee.Utils.HTTP

  @huggingface_endpoint "https://huggingface.co"

  @doc """
  Returns a URL pointing to the given file in a Hugging Face repository.
  """
  @spec file_url(String.t(), String.t(), String.t() | nil) :: String.t()
  def file_url(repository_id, filename, revision) do
    revision = revision || "main"
    @huggingface_endpoint <> "/#{repository_id}/resolve/#{revision}/#{filename}"
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

    * `:offline` - if `true`, cached path is returned if exists and
      and error otherwise

    * `:auth_token` - the token to use as HTTP bearer authorization
      for remote files

  """
  @spec cached_download(String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def cached_download(url, opts \\ []) do
    cache_dir = opts[:cache_dir] || bumblebee_cache_dir()
    offline = opts[:offline] || bumblebee_offline?()
    auth_token = opts[:auth_token]

    dir = Path.join(cache_dir, "huggingface")

    File.mkdir_p!(dir)

    headers =
      if auth_token do
        [{"Authorization", "Bearer " <> auth_token}]
      else
        []
      end

    metadata_path = Path.join(dir, metadata_filename(url))

    if offline do
      case load_json(metadata_path) do
        {:ok, %{"etag" => etag}} ->
          entry_path = Path.join(dir, entry_filename(url, etag))
          {:ok, entry_path}

        _ ->
          {:error, "could not find file in local cache and outgoing traffic is disabled"}
      end
    else
      with {:ok, etag, download_url} <- head_download(url, headers) do
        entry_path = Path.join(dir, entry_filename(url, etag))

        case load_json(metadata_path) do
          {:ok, %{"etag" => ^etag}} ->
            {:ok, entry_path}

          _ ->
            case HTTP.download(download_url, entry_path, headers: headers) |> finish_request() do
              :ok ->
                :ok = store_json(metadata_path, %{"etag" => etag, "url" => url})
                {:ok, entry_path}

              error ->
                File.rm_rf!(metadata_path)
                File.rm_rf!(entry_path)
                error
            end
        end
      end
    end
  end

  defp head_download(url, headers) do
    with {:ok, response} <-
           HTTP.request(:head, url, follow_redirects: false, headers: headers) |> finish_request(),
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

  defp bumblebee_cache_dir() do
    if dir = System.get_env("BUMBLEBEE_CACHE_DIR") do
      Path.expand(dir)
    else
      :filename.basedir(:user_cache, "bumblebee")
    end
  end

  defp bumblebee_offline?() do
    System.get_env("BUMBLEBEE_OFFLINE") in ~w(1 true)
  end
end
