defmodule Bumblebee.Audio.SpeechRecognition do
  @moduledoc false

  alias Bumblebee.Shared

  def speech_recognition(model_info, featurizer, tokenizer, opts \\ []) do
    {compile, opts} = Keyword.pop(opts, :compile)
    {defn_options, opts} = Keyword.pop(opts, :defn_options, [])

    batch_size = compile[:batch_size]

    if compile != nil and batch_size == nil do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size, got: #{inspect(compile)}"
    end

    sampling_rate = featurizer.sampling_rate

    %{model: model, params: params, spec: spec} = model_info

    generate_fun = Bumblebee.Text.Generation.build_generate(model, spec, opts)

    Nx.Serving.new(
      fn ->
        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_features" => Shared.input_template(spec, "input_features", [batch_size])
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          generate_fun.(params, inputs)
        end
      end,
      batch_size: batch_size
    )
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} =
        Shared.validate_serving_input!(input, fn
          path when is_binary(path) ->
            ffmpeg_read_as_pcm(path, sampling_rate)

          %Nx.Tensor{shape: {_, _}} = input ->
            {:ok, input}

          other ->
            {:error, "expected a 2-dimensional tensor or a file path, got: #{inspect(other)}"}
        end)

      inputs = Bumblebee.apply_featurizer(featurizer, inputs, defn_options: defn_options)
      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn token_ids, _metadata, multi? ->
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      decoded
      |> Enum.map(&%{results: [%{text: normalize_text(&1)}]})
      |> Shared.normalize_output(multi?)
    end)
  end

  defp ffmpeg_read_as_pcm(path, sampling_rate) do
    channels = 1

    format =
      case System.endianness() do
        :little -> "f32le"
        :big -> "f32be"
      end

    cond do
      System.find_executable("ffmpeg") == nil ->
        {:error, "ffmpeg not found in PATH"}

      not File.exists?(path) ->
        {:error, "no file found at #{path}"}

      true ->
        System.cmd("ffmpeg", [
          "-i",
          path,
          "-ac",
          Integer.to_string(channels),
          "-ar",
          Integer.to_string(sampling_rate),
          "-f",
          format,
          "-hide_banner",
          "-loglevel",
          "quiet",
          "pipe:1"
        ])
        |> case do
          {data, 0} ->
            input = data |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 1})
            {:ok, input}

          {_, 1} ->
            {:error, "ffmpeg failed to decode the given file"}
        end
    end
  end

  defp normalize_text(text) do
    String.trim(text)
  end
end
