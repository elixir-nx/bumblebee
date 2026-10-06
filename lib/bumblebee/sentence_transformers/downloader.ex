defmodule Bumblebee.SentenceTransformers.Downloader do
  @moduledoc false

  alias Bumblebee.HuggingFace.Hub

  def subrepository(repository, path) when path in [nil, ""], do: repository
  def subrepository({:local, dir}, path), do: {:local, Path.join(dir, path)}

  def subrepository({:local, dir, opts}, path) do
    {:local, Path.join([dir, opts[:subdir] || "", path])}
  end

  def subrepository({:hf, id}, path), do: {:hf, id, subdir: path}

  def subrepository({:hf, id, opts}, path) do
    {:hf, id, Keyword.put(opts, :subdir, Path.join(opts[:subdir] || "", path))}
  end

  @doc """
  Downloads or loads a specific file from a repository on demand.
  """
  def download_file(repository, filename, opts \\ [])

  def download_file({:local, dir}, filename, _opts) when is_binary(dir) do
    check_local_file(Path.join(dir, filename))
  end

  def download_file({:local, dir, local_opts}, filename, _opts) when is_binary(dir) do
    subdir = local_opts[:subdir] || ""
    check_local_file(Path.join([dir, subdir, filename]))
  end

  def download_file({:hf, repo_id}, filename, opts) do
    download_file({:hf, repo_id, []}, filename, opts)
  end

  def download_file({:hf, repo_id, repo_opts}, filename, opts) do
    merged_opts = Keyword.merge(repo_opts, opts)
    subdir = merged_opts[:subdir]

    full_filename =
      if subdir do
        subdir <> "/" <> filename
      else
        filename
      end

    url = Hub.file_url(repo_id, full_filename, merged_opts[:revision])
    cache_scope = repo_id_to_cache_scope(repo_id)

    Hub.cached_download(
      url,
      [cache_scope: cache_scope] ++ Keyword.take(merged_opts, [:cache_dir, :offline, :auth_token])
    )
  end

  def download_file(other, _filename, _opts) do
    {:error, "invalid repository format: #{inspect(other)}"}
  end

  defp check_local_file(path) do
    if File.exists?(path) do
      {:ok, path}
    else
      {:error, "file #{inspect(path)} does not exist"}
    end
  end

  @doc false
  def missing_file?(repository, filename, reason) do
    case repository do
      {:local, dir} -> missing_local_file?(Path.join(dir, filename))
      {:local, dir, opts} -> missing_local_file?(Path.join([dir, opts[:subdir] || "", filename]))
      _ -> is_binary(reason) and String.starts_with?(reason, "file not found, url: ")
    end
  end

  defp missing_local_file?(path) do
    case File.stat(path) do
      {:error, :enoent} -> true
      _ -> false
    end
  end

  @doc """
  Checks whether a file exists in the repository.
  """
  def file_exists?(repository, filename, opts \\ []) do
    case download_file(repository, filename, opts) do
      {:ok, _path} -> true
      {:error, _} -> false
    end
  end

  defp repo_id_to_cache_scope(repo_id) do
    repo_id
    |> String.replace("/", "--")
    |> String.replace(~r/[^\w-]/, "")
  end
end
