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
    opts =
      Keyword.validate!(opts, [
        :chunk_num_seconds,
        :context_num_seconds,
        :seed,
        :compile,
        defn_options: [],
        preallocate_params: false
      ])

    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [:for_conditional_generation])

    chunk_num_seconds = opts[:chunk_num_seconds]
    context_num_seconds = opts[:context_num_seconds]
    preallocate_params = opts[:preallocate_params]
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
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

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

      all_chunks =
        for input <- inputs do
          if chunk_num_seconds do
            chunk_input(input, sampling_rate, chunk_num_seconds, context_num_seconds)
          else
            [input]
          end
        end

      all_num_chunks = Enum.map(all_chunks, &length/1)

      all_chunks = List.flatten(all_chunks)
      inputs = Bumblebee.apply_featurizer(featurizer, all_chunks, defn_options: defn_options)
      {Nx.Batch.concatenate([inputs]), {multi?, all_num_chunks}}
    end)
    |> Nx.Serving.client_postprocessing(fn {results, _metadata}, {multi?, all_num_chunks} ->
      all_special_tokens = Bumblebee.Tokenizer.all_special_tokens(tokenizer)

      sequences =
        results
        |> Bumblebee.Utils.Nx.to_list()
        |> Enum.map(fn sequence ->
          sequence
          |> Enum.filter(fn token_id ->
            if token = Bumblebee.Tokenizer.id_to_token(tokenizer, token_id) do
              token not in all_special_tokens
            end
          end)
          |> Nx.tensor()
        end)

      {outputs, []} =
        Enum.map_reduce(all_num_chunks, sequences, fn num_chunks, sequences ->
          {sequences, rest} = Enum.split(sequences, num_chunks)
          token_ids = merge_overlapping_sequences(sequences)
          text = Bumblebee.Tokenizer.decode(tokenizer, token_ids)
          output = %{results: [%{text: normalize_text(text)}]}
          {output, rest}
        end)

      Shared.normalize_output(outputs, multi?)
    end)
  end

  defp chunk_input(input, sampling_rate, chunk_num_seconds, context_num_seconds) do
    context_num_seconds = context_num_seconds || chunk_num_seconds / 6

    chunk_length = floor(chunk_num_seconds * sampling_rate)
    context_left = floor(context_num_seconds * sampling_rate)
    context_right = context_left

    input_length = Nx.axis_size(input, 0)
    step = chunk_length - context_left - context_right

    0..(input_length - 1)//step
    |> Enum.reduce_while([], fn chunk_start_idx, chunks ->
      chunk_end_idx = chunk_start_idx + chunk_length

      # All right contexts must be full, otherwise it is the last item
      last? =
        if context_right > 0 do
          chunk_end_idx > input_length
        else
          chunk_end_idx >= input_length
        end

      chunk = input[chunk_start_idx..(min(chunk_end_idx, input_length) - 1)]
      chunks = [chunk | chunks]

      {if(last?, do: :halt, else: :cont), chunks}
    end)
    |> Enum.reverse()
  end

  defp merge_overlapping_sequences(sequences) do
    # We have a number of consecutive, overlapping sequences and we
    # want to merge them into a single sequence. To merge a pair of
    # consecutive sequences we slide the sequences and compare the
    # overlap:
    #
    #     abcd    (left)
    #        cde  (right)
    #     => compare c = d
    #
    #     abcd    (left)
    #       cde   (right)
    #     => compare cd = cd
    #
    # We find the best alignment, then cut the overlap in half and
    # concatenate the left an right part accordingly. In the example
    # above, we would use the second alignment, taking `abc` from the
    # left sequence and `de` from the right one.

    {[left_sequence], right_sequences} = Enum.split(sequences, 1)

    {acc, left_sequence} =
      for right_sequence <- right_sequences, reduce: {[], left_sequence} do
        {acc, left_sequence} ->
          left_length = Nx.size(left_sequence)
          right_length = Nx.size(right_sequence)

          {_max_match_score, overlap_indices} =
            for i <- 1..(left_length + right_length - 1),
                reduce: {0.0, {left_length, left_length, 0, 0}} do
              {max_match_score, overlap_indices} ->
                left_start = max(0, left_length - i)
                left_stop = min(left_length, left_length + right_length - i)
                left_overlap = left_sequence[left_start..(left_stop - 1)]

                right_start = max(0, i - left_length)
                right_stop = min(right_length, i)
                right_overlap = right_sequence[right_start..(right_stop - 1)]

                num_matches = Nx.equal(left_overlap, right_overlap) |> Nx.sum() |> Nx.to_number()

                # Epsilon to favor long perfect matches
                eps = i / 10000.0
                match_score = num_matches / i + eps

                if num_matches > 1 and match_score > max_match_score do
                  overlap_indices = {left_start, left_stop, right_start, right_stop}
                  {match_score, overlap_indices}
                else
                  {max_match_score, overlap_indices}
                end
            end

          # Cut in the middle of the overlap
          {left_start, left_stop, right_start, right_stop} = overlap_indices
          left_mid = div(left_stop + left_start, 2)
          right_mid = div(right_stop + right_start, 2)
          {[left_sequence[0..(left_mid - 1)] | acc], right_sequence[right_mid..-1//1]}
      end

    Enum.reduce([left_sequence | acc], [], fn sequence, acc ->
      Nx.to_flat_list(sequence) ++ acc
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
