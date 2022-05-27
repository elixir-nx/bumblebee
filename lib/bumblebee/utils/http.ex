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
  @spec download(String.t(), Path.t(), keyword()) :: :ok | {:error, term()}
  def download(url, path, opts \\ []) do
    path = IO.chardata_to_string(path)
    headers = build_headers(opts[:headers] || [])

    request = {url, headers}
    http_opts = [ssl: http_ssl_opts()]
    opts = [stream: String.to_charlist(path)]

    case :httpc.request(:get, request, http_opts, opts) do
      {:ok, :saved_to_file} ->
        :ok

      {:ok, {{_, 200, _}, _headers, body}} ->
        File.write(path, body)

      {:ok, {{_, status, _}, _headers, _body}} ->
        {:error, "download failed, got HTTP status: #{status}"}

      {:error, error} ->
        {:error, error}
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
  @spec request(atom(), String.t(), keyword()) :: {:ok, response()} | {:error, term()}
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
        {:error, error}
    end
  end

  defp build_headers(entries) do
    headers =
      Enum.map(entries, fn {key, value} ->
        {to_charlist(key), to_charlist(value)}
      end)

    [{'user-agent', 'bumblebee'} | headers]
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
