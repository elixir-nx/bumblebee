defmodule Bumblebee.Utils.HTTP do
  @moduledoc false

  @type response :: %{status: status(), headers: headers(), body: binary()}
  @type status :: non_neg_integer()
  @type headers :: list(header())
  @type header :: {String.t(), String.t()}

  @doc """
  Retrieves the header value from response headers.
  """
  @spec get_header(response(), String.t()) :: String.t() | nil
  def get_header(response, key) do
    with {^key, value} <- List.keyfind(response.headers, key, 0), do: value
  end

  @doc """
  Downloads resource at the given URL to a file.

  ## Options

    * `:headers` - request headers

  """
  @spec download(String.t(), Path.t(), keyword()) :: :ok | {:error, String.t()}
  def download(url, path, opts \\ []) do
    path = IO.chardata_to_string(path)
    headers = build_headers(opts[:headers] || [])

    case File.open(path, [:write]) do
      {:ok, file} ->
        try do
          request = {url, headers}
          http_opts = [ssl: http_ssl_opts()]

          caller = self()

          receiver = fn reply_info ->
            request_id = elem(reply_info, 0)

            # Cancel the request if the caller terminates
            if Process.alive?(caller) do
              send(caller, {:http, reply_info})
            else
              :httpc.cancel_request(request_id)
            end
          end

          opts = [stream: :self, sync: false, receiver: receiver]

          {:ok, request_id} = :httpc.request(:get, request, http_opts, opts)
          download_loop(%{request_id: request_id, file: file, total_size: nil, size: nil})
        after
          File.close(file)
        end

      {:error, error} ->
        {:error, "failed to open file for download, reason: #{:file.format_error(error)}"}
    end
  end

  defp download_loop(state) do
    receive do
      {:http, reply_info} when elem(reply_info, 0) == state.request_id ->
        download_receive(state, reply_info)
    end
  end

  defp download_receive(_state, {_, {:error, error}}) do
    {:error, "download failed, reason: #{inspect(error)}"}
  end

  defp download_receive(state, {_, {{_, 200, _}, _headers, body}}) do
    case IO.binwrite(state.file, body) do
      :ok ->
        :ok

      {:error, error} ->
        {:error, "failed to write to file, reason: #{:file.format_error(error)}"}
    end
  end

  defp download_receive(_state, {_, {{_, status, _}, _headers, _body}}) do
    {:error, "download failed, got HTTP status: #{status}"}
  end

  defp download_receive(state, {_, :stream_start, headers}) do
    total_size = total_size(headers)
    download_loop(%{state | total_size: total_size, size: 0})
  end

  defp download_receive(state, {_, :stream, body_part}) do
    case IO.binwrite(state.file, body_part) do
      :ok ->
        part_size = byte_size(body_part)
        state = update_in(state.size, &(&1 + part_size))

        if Bumblebee.Utils.progress_bar_enabled?() &&
             state.total_size && part_size != state.total_size do
          ProgressBar.render(state.size, state.total_size, suffix: :bytes)
        end

        download_loop(state)

      {:error, error} ->
        :httpc.cancel_request(state.request_id)
        {:error, "failed to write to file, reason: #{:file.format_error(error)}"}
    end
  end

  defp download_receive(_state, {_, :stream_end, _headers}) do
    :ok
  end

  defp total_size(headers) do
    case List.keyfind(headers, ~c"content-length", 0) do
      {_, content_length} ->
        content_length |> List.to_string() |> String.to_integer()

      _ ->
        nil
    end
  end

  @doc """
  Makes an HTTP request.

  ## Options

    * `:headers` - request headers

    * `:body` - request body given as `{content_type, body}`

    * `:timeout` - request timeout in milliseconds. Defaults to `10_000`

    * `:follow_redirects` - whether to automatically repeat the request
      to the redirect location. Defaults to `true`

  """
  @spec request(atom(), String.t(), keyword()) :: {:ok, response()} | {:error, String.t()}
  def request(method, url, opts \\ []) do
    headers = build_headers(opts[:headers] || [])
    follow_redirects = Keyword.get(opts, :follow_redirects, true)

    request =
      case opts[:body] do
        nil -> {url, headers}
        {content_type, body} -> {url, headers, to_charlist(content_type), body}
      end

    http_opts = [
      ssl: http_ssl_opts(),
      timeout: opts[:timeout] || 10_000,
      autoredirect: follow_redirects
    ]

    opts = [
      body_format: :binary
    ]

    case :httpc.request(method, request, http_opts, opts) do
      {:ok, {{_, status, _}, headers, body}} ->
        {:ok, %{status: status, headers: parse_headers(headers), body: body}}

      {:error, error} ->
        {:error, "HTTP request failed, reason: #{inspect(error)}"}
    end
  end

  defp build_headers(entries) do
    headers =
      Enum.map(entries, fn {key, value} ->
        {to_charlist(key), to_charlist(value)}
      end)

    [{~c"user-agent", ~c"bumblebee"} | headers]
  end

  defp parse_headers(headers) do
    Enum.map(headers, fn {key, val} ->
      {String.downcase(to_string(key)), to_string(val)}
    end)
  end

  defp http_ssl_opts() do
    # Use secure options, see https://gist.github.com/jonatanklosko/5e20ca84127f6b31bbe3906498e1a1d7
    [
      verify: :verify_peer,
      cacertfile: CAStore.file_path(),
      customize_hostname_check: [
        match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
      ]
    ]
  end
end
