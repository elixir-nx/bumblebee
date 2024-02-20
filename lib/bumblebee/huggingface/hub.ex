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
  Returns a URL to list the contents of a Hugging Face repository.
  """
  @spec file_listing_url(String.t(), String.t() | nil, String.t() | nil) :: String.t()
  def file_listing_url(repository_id, subdir, revision) do
    revision = revision || "main"
    path = if(subdir, do: "/" <> subdir)
    @huggingface_endpoint <> "/api/models/#{repository_id}/tree/#{revision}#{path}"
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

    * `:etag` - by default a HEAD request is made to fetch the latest
      ETag value, however if the value is already known, it can be
      passed as an option instead (to skip the extra request)

    * `:cache_scope` - a namespace to put the cached files under in
      the cache directory

  """
  @spec cached_download(String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def cached_download(url, opts \\ []) do
    cache_dir = opts[:cache_dir] || Bumblebee.cache_dir()
    offline = Keyword.get(opts, :offline, bumblebee_offline?())
    auth_token = opts[:auth_token]

    dir = Path.join(cache_dir, "huggingface")

    dir =
      if cache_scope = opts[:cache_scope] do
        Path.join(dir, cache_scope)
      else
        dir
      end

    File.mkdir_p!(dir)

    headers =
      if auth_token do
        [{"Authorization", "Bearer " <> auth_token}]
      else
        []
      end

    metadata_path = Path.join(dir, metadata_filename(url))

    cond do
      offline ->
        case load_json(metadata_path) do
          {:ok, %{"etag" => etag}} ->
            entry_path = Path.join(dir, entry_filename(url, etag))
            {:ok, entry_path}

          _ ->
            {:error,
             "could not find file in local cache and outgoing traffic is disabled, url: #{url}"}
        end

      entry_path = opts[:etag] && cached_path_for_etag(dir, url, opts[:etag]) ->
        {:ok, entry_path}

      true ->
        with {:ok, etag, download_url, redirect?} <- head_download(url, headers) do
          if entry_path = cached_path_for_etag(dir, url, etag) do
            {:ok, entry_path}
          else
            entry_path = Path.join(dir, entry_filename(url, etag))

            headers =
              if redirect? do
                List.keydelete(headers, "Authorization", 0)
              else
                headers
              end

            download_url
            |> HTTP.download(entry_path, headers: headers)
            |> finish_request(download_url)
            |> case do
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

  defp cached_path_for_etag(dir, url, etag) do
    metadata_path = Path.join(dir, metadata_filename(url))

    case load_json(metadata_path) do
      {:ok, %{"etag" => ^etag}} ->
        path = Path.join(dir, entry_filename(url, etag))

        # Make sure the file exists, in case someone manually removed it
        if File.exists?(path) do
          path
        end

      _ ->
        nil
    end
  end

  defp head_download(url, headers) do
    with {:ok, response} <-
           HTTP.request(:head, url, follow_redirects: false, headers: headers)
           |> finish_request(url) do
      if response.status in 300..399 do
        location = HTTP.get_header(response, "location")

        # Follow relative redirects
        if URI.parse(location).host == nil do
          url =
            url
            |> URI.parse()
            |> Map.replace!(:path, location)
            |> URI.to_string()

          head_download(url, headers)
        else
          with {:ok, etag} <- fetch_etag(response), do: {:ok, etag, location, true}
        end
      else
        with {:ok, etag} <- fetch_etag(response), do: {:ok, etag, url, false}
      end
    end
  end

  defp finish_request(:ok, _url), do: :ok

  defp finish_request({:ok, response}, _url) when response.status in 100..399, do: {:ok, response}

  defp finish_request({:ok, response}, url) do
    case HTTP.get_header(response, "x-error-code") do
      code when code == "RepoNotFound" or response.status == 401 ->
        {:error,
         "repository not found, url: #{url}. Please make sure you specified" <>
           " the correct repository id. If you are trying to access a private" <>
           " or gated repository, use an authentication token"}

      "EntryNotFound" ->
        {:error, "file not found, url: #{url}"}

      "RevisionNotFound" ->
        {:error, "revision not found, url: #{url}"}

      "GatedRepo" ->
        {:error,
         "cannot access gated repository, url: #{url}. Make sure to request access" <>
           " for the repository and use an authentication token"}

      _ ->
        {:error, "HTTP request failed with status #{response.status}, url: #{url}"}
    end
  end

  defp finish_request({:error, reason}, _url) do
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

  defp bumblebee_offline?() do
    System.get_env("BUMBLEBEE_OFFLINE") in ~w(1 true)
  end
end
