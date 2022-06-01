defmodule Bumblebee.HuggingFace.HubTest do
  use ExUnit.Case, async: true

  alias Bumblebee.HuggingFace.Hub

  setup do
    bypass = Bypass.open()
    {:ok, bypass: bypass}
  end

  describe "cached_download/2" do
    @tag :tmp_dir
    test "checks etag and downloads the file", %{bypass: bypass, tmp_dir: tmp_dir} do
      Bypass.expect_once(bypass, "HEAD", "/file.json", fn conn ->
        serve_with_etag(conn, ~s/"hash"/, "")
      end)

      Bypass.expect_once(bypass, "GET", "/file.json", fn conn ->
        serve_with_etag(conn, ~s/"hash"/, "{}")
      end)

      url = url(bypass.port) <> "/file.json"

      assert {:ok, path} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert File.read!(path) == "{}"
    end

    @tag :tmp_dir
    test "returns a cached file when etag matches", %{bypass: bypass, tmp_dir: tmp_dir} do
      Bypass.expect(bypass, "HEAD", "/file.json", fn conn ->
        serve_with_etag(conn, ~s/"hash"/, "")
      end)

      Bypass.expect_once(bypass, "GET", "/file.json", fn conn ->
        serve_with_etag(conn, ~s/"hash"/, "{}")
      end)

      url = url(bypass.port) <> "/file.json"

      assert {:ok, path} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert {:ok, ^path} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert File.read!(path) == "{}"
    end

    @tag :tmp_dir
    test "redownloads the file when etag changes", %{bypass: bypass, tmp_dir: tmp_dir} do
      counter = start_supervised!({Agent, fn -> 0 end})

      Bypass.expect(bypass, "HEAD", "/file.json", fn conn ->
        case Agent.get(counter, & &1) do
          0 -> serve_with_etag(conn, ~s/"hash"/, "")
          1 -> serve_with_etag(conn, ~s/"hash2"/, "[]")
        end
      end)

      Bypass.expect(bypass, "GET", "/file.json", fn conn ->
        counter
        |> Agent.get_and_update(fn counter -> {counter, counter + 1} end)
        |> case do
          0 -> serve_with_etag(conn, ~s/"hash"/, "{}")
          1 -> serve_with_etag(conn, ~s/"hash2"/, "[]")
        end
      end)

      url = url(bypass.port) <> "/file.json"

      assert {:ok, path1} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert {:ok, path2} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert path1 != path2
      assert File.read!(path2) == "[]"
    end

    @tag :tmp_dir
    test "handles redirected downloads", %{bypass: bypass, tmp_dir: tmp_dir} do
      Bypass.expect_once(bypass, "HEAD", "/file.bin", fn conn ->
        url = Plug.Conn.request_url(conn)

        conn
        |> Plug.Conn.put_resp_header("x-linked-etag", ~s/"hash"/)
        |> Plug.Conn.put_resp_header("location", url <> "/storage")
        |> Plug.Conn.resp(302, "")
      end)

      Bypass.expect_once(bypass, "GET", "/file.bin/storage", fn conn ->
        conn
        |> Plug.Conn.put_resp_header("etag", ~s/"hash"/)
        |> Plug.Conn.resp(200, <<0, 1>>)
      end)

      url = url(bypass.port) <> "/file.bin"

      assert {:ok, path} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert File.read!(path) == <<0, 1>>
    end

    @tag :tmp_dir
    test "caches redirected downloads based on x-linked-etag", %{bypass: bypass, tmp_dir: tmp_dir} do
      Bypass.expect(bypass, "HEAD", "/file.bin", fn conn ->
        url = Plug.Conn.request_url(conn)

        conn
        |> Plug.Conn.put_resp_header("x-linked-etag", ~s/"hash"/)
        |> Plug.Conn.put_resp_header("location", url <> "/storage")
        |> Plug.Conn.resp(302, "")
      end)

      Bypass.expect_once(bypass, "GET", "/file.bin/storage", fn conn ->
        serve_with_etag(conn, ~s/"hash"/, <<0, 1>>)
      end)

      url = url(bypass.port) <> "/file.bin"

      assert {:ok, path} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert {:ok, ^path} = Hub.cached_download(url, cache_dir: tmp_dir)
      assert File.read!(path) == <<0, 1>>
    end

    @tag :tmp_dir
    test "returns an error on missing etag header", %{bypass: bypass, tmp_dir: tmp_dir} do
      Bypass.expect_once(bypass, "HEAD", "/file.json", fn conn ->
        Plug.Conn.resp(conn, 200, "")
      end)

      url = url(bypass.port) <> "/file.json"

      assert {:error, "no ETag found on the resource"} =
               Hub.cached_download(url, cache_dir: tmp_dir)
    end

    @tag :tmp_dir
    test "returns an error on http failure", %{bypass: bypass, tmp_dir: tmp_dir} do
      Bypass.expect_once(bypass, "HEAD", "/file.json", fn conn ->
        Plug.Conn.resp(conn, 500, "")
      end)

      url = url(bypass.port) <> "/file.json"

      assert {:error, "HTTP request failed with status 500"} =
               Hub.cached_download(url, cache_dir: tmp_dir)
    end

    @tag :tmp_dir
    test "returns more specific error if x-error-code is present",
         %{bypass: bypass, tmp_dir: tmp_dir} do
      Bypass.expect_once(bypass, "HEAD", "/file.json", fn conn ->
        conn
        |> Plug.Conn.put_resp_header("x-error-code", "RepoNotFound")
        |> Plug.Conn.resp(404, "")
      end)

      url = url(bypass.port) <> "/file.json"

      assert {:error, "repository not found"} = Hub.cached_download(url, cache_dir: tmp_dir)
    end
  end

  defp url(port), do: "http://localhost:#{port}"

  defp serve_with_etag(conn, etag, body) do
    conn
    |> Plug.Conn.put_resp_header("etag", etag)
    |> Plug.Conn.resp(200, body)
  end
end
