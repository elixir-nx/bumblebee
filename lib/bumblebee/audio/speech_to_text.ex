defmodule Bumblebee.Audio.SpeechToText do
  @moduledoc false

  alias Bumblebee.Shared
  alias Bumblebee.Text

  def speech_to_text(
        model_info,
        featurizer,
        tokenizer,
        %Text.GenerationConfig{} = generation_config,
        opts \\ []
      ) do
    opts = Keyword.validate!(opts, [:seed, :compile, defn_options: []])

    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [:for_conditional_generation])

    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size])
        |> Shared.require_options!([:batch_size])
      end

    batch_size = compile[:batch_size]

    sampling_rate = featurizer.sampling_rate

    generate_fun =
      Text.Generation.build_generate(model, spec, generation_config, Keyword.take(opts, [:seed]))

    Nx.Serving.new(
      fn defn_options ->
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
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} =
        Shared.validate_serving_input!(input, fn
          %Nx.Tensor{shape: {_}} = input ->
            {:ok, input}

          {:file, path} when is_binary(path) ->
            ffmpeg_read_as_pcm(path, sampling_rate)

          other ->
            {:error, "expected a 1-dimensional tensor or {:file, path}, got: #{inspect(other)}"}
        end)

      inputs = Bumblebee.apply_featurizer(featurizer, inputs, defn_options: defn_options)
      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {token_ids, _metadata}, multi? ->
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
            {:ok, Nx.from_binary(data, :f32)}

          {_, 1} ->
            {:error, "ffmpeg failed to decode the given file"}
        end
    end
  end

  defp normalize_text(text) do
    String.trim(text)
  end
end
