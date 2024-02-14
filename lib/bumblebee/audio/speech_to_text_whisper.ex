defmodule Bumblebee.Audio.SpeechToTextWhisper do
  @moduledoc false

  alias Bumblebee.Shared
  alias Bumblebee.Text

  def speech_to_text_whisper(
        model_info,
        featurizer,
        tokenizer,
        %Text.GenerationConfig{} = generation_config,
        opts \\ []
      )
      when is_struct(model_info.spec, Bumblebee.Audio.Whisper) do
    opts =
      Keyword.validate!(opts, [
        :chunk_num_seconds,
        :context_num_seconds,
        :language,
        :compile,
        :timestamps,
        defn_options: [],
        preallocate_params: false,
        task: :transcribe,
        stream: false
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
    timestamps? = opts[:timestamps] != nil

    {generate_opts, generation_config} = generate_opts(generation_config, opts)
    generate_fun = Text.Generation.build_generate(model, spec, generation_config, generate_opts)

    generate_fun = fn params, {inputs, seed} ->
      inputs = Bumblebee.Featurizer.process_batch(featurizer, inputs)
      inputs = Map.put(inputs, "seed", seed)
      %{token_ids: token_ids} = generate_fun.(params, inputs)
      token_ids
    end

    Nx.Serving.new(
      fn defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            inputs = Bumblebee.Featurizer.batch_template(featurizer, batch_size)
            seed = Nx.template({batch_size}, :s64)
            [params, {inputs, seed}]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          generate_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      if opts[:stream] do
        Shared.validate_input_for_stream!(input)
      end

      {inputs, multi?} = Shared.validate_serving_input!(input, &validate_input(&1, sampling_rate))

      all_chunks =
        for input <- inputs do
          if chunk_num_seconds do
            chunks =
              chunk_input(input.audio, sampling_rate, chunk_num_seconds, context_num_seconds)

            for {chunk, lengths} <- chunks, do: {{chunk, input.seed}, lengths}
          else
            [{{input.audio, input.seed}, nil}]
          end
        end

      all_num_chunks = Enum.map(all_chunks, &length/1)

      all_chunks = List.flatten(all_chunks)
      {all_chunks, lengths} = Enum.unzip(all_chunks)

      if batch_size do
        stream =
          all_chunks
          |> Stream.chunk_every(batch_size)
          |> Stream.map(fn all_chunks ->
            {all_chunks, seed} = Enum.unzip(all_chunks)
            seed = Nx.tensor(seed, backend: Nx.BinaryBackend)
            inputs = Bumblebee.Featurizer.process_input(featurizer, all_chunks)
            Nx.Batch.concatenate([{inputs, seed}])
          end)

        {stream, {multi?, all_num_chunks, lengths}}
      else
        {all_chunks, seed} = Enum.unzip(all_chunks)
        seed = Nx.tensor(seed, backend: Nx.BinaryBackend)
        inputs = Bumblebee.Featurizer.process_input(featurizer, all_chunks)
        {Nx.Batch.concatenate([{inputs, seed}]), {multi?, all_num_chunks, lengths}}
      end
    end)
    |> maybe_stream(opts[:stream], spec, featurizer, tokenizer, timestamps?)
  end

  defp validate_input(%{audio: audio} = input, sampling_rate) do
    with {:ok, audio} <- parse_audio(audio, sampling_rate) do
      {:ok, %{audio: audio, seed: input[:seed] || :erlang.system_time()}}
    end
  end

  defp validate_input(input, sampling_rate), do: validate_input(%{audio: input}, sampling_rate)

  defp parse_audio(input, sampling_rate) do
    case input do
      %Nx.Tensor{shape: {_}} = input ->
        {:ok, Nx.backend_transfer(input, Nx.BinaryBackend)}

      {:file, path} when is_binary(path) ->
        ffmpeg_read_as_pcm(path, sampling_rate)

      other ->
        {:error,
         "expected audio to be a 1-dimensional tensor or {:file, path}, got: #{inspect(other)}"}
    end
  end

  defp maybe_stream(serving, false, spec, featurizer, tokenizer, timestamps?) do
    Nx.Serving.client_postprocessing(serving, fn
      {outputs, _metadata}, {multi?, all_num_chunks, lengths} ->
        chunk_outputs = Nx.to_list(outputs)

        all_num_chunks
        |> Enum.map_reduce(chunk_outputs, fn num_chunks, chunk_outputs ->
          {outputs, rest} = Enum.split(chunk_outputs, num_chunks)
          state = decode_chunk_outputs_init(lengths, spec, featurizer, tokenizer)
          {chunks, _state} = decode_chunk_outputs_update(state, outputs, timestamps?, tokenizer)
          {%{chunks: chunks}, rest}
        end)
        |> elem(0)
        |> Shared.normalize_output(multi?)
    end)
  end

  defp maybe_stream(serving, true, spec, featurizer, tokenizer, timestamps?) do
    serving
    |> Nx.Serving.streaming()
    |> Nx.Serving.client_postprocessing(fn stream, {false = _multi?, _all_num_chunks, lengths} ->
      state = decode_chunk_outputs_init(lengths, spec, featurizer, tokenizer)

      Stream.transform(stream, state, fn {:batch, outputs, _metadata}, state ->
        outputs = Nx.to_list(outputs)
        decode_chunk_outputs_update(state, outputs, timestamps?, tokenizer)
      end)
    end)
  end

  defp generate_opts(generation_config, opts) do
    forced_token_ids = forced_token_ids(opts, generation_config.extra_config)
    generation_config = %{generation_config | forced_token_ids: forced_token_ids}

    logits_processors =
      if opts[:timestamps] do
        [
          &Bumblebee.Text.Generation.LogitsProcessing.whisper_timestamp_processor(&1, &2,
            eos_token_id: generation_config.eos_token_id,
            forced_token_ids: generation_config.forced_token_ids,
            no_timestamps_token_id: generation_config.extra_config.no_timestamps_token_id,
            timestamp_begin_id: generation_config.extra_config.no_timestamps_token_id + 1
          )
        ]
      else
        []
      end

    opts = [logits_processors: logits_processors]

    {opts, generation_config}
  end

  defp forced_token_ids(opts, extra_config) do
    token_ids =
      if language = opts[:language] do
        if extra_config.task_to_token_id == %{} do
          raise "the generation config does not have any languages defined." <>
                  " If you are dealing with a monolingual model, set :language to nil." <>
                  " Otherwise you may need to update generation_config.extra_config.language_to_token_id"
        end

        language_token_id = extra_config.language_to_token_id[language]

        unless language_token_id do
          values =
            extra_config.language_to_token_id
            |> Map.keys()
            |> Enum.sort()
            |> Enum.map_join(", ", &inspect/1)

          raise "invalid language #{inspect(language)}, expected one of: #{values}"
        end

        [language_token_id]
      else
        [nil]
      end ++
        if task = opts[:task] do
          if extra_config.task_to_token_id == %{} do
            raise "the generation config does not have any tasks defined." <>
                    " If you are dealing with a monolingual model, set :task to nil." <>
                    " Otherwise you may need to update generation_config.extra_config.task_to_token_id"
          end

          task_token_id = extra_config.task_to_token_id[task]

          unless task_token_id do
            values =
              extra_config.task_to_token_id
              |> Map.keys()
              |> Enum.sort()
              |> Enum.map_join(", ", &inspect/1)

            raise "invalid task #{inspect(task)}, expected one of: #{values}"
          end

          [task_token_id]
        else
          []
        end ++
        if opts[:timestamps] do
          []
        else
          [extra_config.no_timestamps_token_id]
        end

    for {token_id, idx} <- Enum.with_index(token_ids, 1), token_id, do: {idx, token_id}
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

      context_left = if chunk_start_idx == 0, do: 0, else: context_left
      context_right = if last?, do: 0, else: context_right

      lengths =
        {chunk_length / sampling_rate, context_left / sampling_rate,
         context_right / sampling_rate}

      chunk = input[chunk_start_idx..(min(chunk_end_idx, input_length) - 1)]
      chunks = [{chunk, lengths} | chunks]

      {if(last?, do: :halt, else: :cont), chunks}
    end)
    |> Enum.reverse()
  end

  # We generalize the decoding into multiple steps, where we feed a
  # number of outputs at a time. When not streaming we just feed all
  # outputs at once.
  defp decode_chunk_outputs_init(lengths, spec, featurizer, tokenizer) do
    time_precision = featurizer.num_seconds / spec.encoder_max_positions
    all_special_tokens = Bumblebee.Tokenizer.all_special_tokens(tokenizer)
    timestamp_begin_id = Bumblebee.Tokenizer.token_to_id(tokenizer, "<|notimestamps|>") + 1

    acc = %{
      time_offset: 0,
      previous_sequences: [],
      current_sequence: [],
      chunks: [],
      chunk: empty_chunk()
    }

    %{
      acc: acc,
      lengths: lengths,
      time_precision: time_precision,
      all_special_tokens: all_special_tokens,
      timestamp_begin_id: timestamp_begin_id
    }
  end

  defp decode_chunk_outputs_update(state, outputs, timestamps?, tokenizer) do
    %{
      time_precision: time_precision,
      timestamp_begin_id: timestamp_begin_id,
      all_special_tokens: all_special_tokens
    } = state

    batch_size = length(outputs)

    {lengths, lengths_rest} = Enum.split(state.lengths, batch_size)
    state = %{state | lengths: lengths_rest}

    acc =
      Enum.zip_reduce([outputs, lengths], state.acc, fn [sequence, lengths], acc ->
        process_output(
          sequence,
          lengths,
          timestamps?,
          timestamp_begin_id,
          time_precision,
          tokenizer,
          all_special_tokens,
          acc
        )
      end)

    acc =
      if timestamps? do
        acc
      else
        # We finish chunks on end timestamps, so with timestamps disabled
        # we need to do this explicitly

        acc = finish_chunk(acc, timestamps?, tokenizer)

        finished? = state.lengths == []

        if finished? do
          # Flush any pending sequences
          finish_chunk(acc, timestamps?, tokenizer)
        else
          acc
        end
      end

    chunks = Enum.reverse(acc.chunks)
    acc = %{acc | chunks: []}
    {chunks, %{state | acc: acc}}
  end

  defp process_output(
         sequence,
         lengths,
         timestamps?,
         timestamp_begin_id,
         time_precision,
         tokenizer,
         all_special_tokens,
         acc
       ) do
    {chunk_length_seconds, context_left_seconds, context_right_seconds} =
      lengths || {nil, 0.0, 0.0}

    time_offset = acc.time_offset - context_left_seconds

    # We want to ignore timestamps in the right and left contexts,
    # because splitting on those would cause issues with merging
    # chunk overlaps. Also note that for right context, if there
    # are not timestamps in the right context, we ignore the last
    # regular timestamp, otherwise we may not have enough tokens
    # to merge chunks properly.

    first_timestamp = timestamp_begin_id + context_left_seconds / time_precision

    last_timestamp =
      if context_right_seconds > 0 do
        right_context_start = chunk_length_seconds - context_right_seconds

        sequence
        |> Enum.reverse()
        |> Enum.reduce_while(nil, fn token_id, last_timestamp ->
          if token_id >= timestamp_begin_id do
            if last_timestamp != nil and
                 (token_id - timestamp_begin_id) * time_precision < right_context_start do
              {:halt, last_timestamp}
            else
              {:cont, token_id}
            end
          else
            {:cont, last_timestamp}
          end
        end)
      end

    acc =
      Enum.reduce(sequence, acc, fn token_id, acc ->
        if token_id >= timestamp_begin_id do
          time = (token_id - timestamp_begin_id) * time_precision + time_offset
          time = Float.round(time, 2)

          cond do
            last_timestamp && token_id >= last_timestamp ->
              acc

            # If we are continuing previously open timestamp, ignore
            # timestamps within the left context
            acc.previous_sequences != [] and token_id < first_timestamp ->
              acc

            acc.chunk.start_timestamp_seconds == nil ->
              put_in(acc.chunk.start_timestamp_seconds, time)

            acc.chunk.start_timestamp_seconds == time ->
              acc

            true ->
              acc = put_in(acc.chunk.end_timestamp_seconds, time)

              acc
              |> finish_current_sequence()
              |> finish_chunk(timestamps?, tokenizer)
          end
        else
          token = Bumblebee.Tokenizer.id_to_token(tokenizer, token_id)

          if token in all_special_tokens do
            # Skip special tokens
            acc
          else
            %{acc | current_sequence: [token_id | acc.current_sequence]}
          end
        end
      end)

    acc = finish_current_sequence(acc)

    time_offset =
      if chunk_length_seconds do
        time_offset + chunk_length_seconds - context_right_seconds
      else
        time_offset
      end

    %{acc | time_offset: time_offset}
  end

  defp finish_current_sequence(%{current_sequence: []} = acc), do: acc

  defp finish_current_sequence(acc) do
    %{
      acc
      | current_sequence: [],
        previous_sequences: [Enum.reverse(acc.current_sequence) | acc.previous_sequences]
    }
  end

  defp finish_chunk(%{previous_sequences: []} = acc, _timestamps?, _tokenizer) do
    %{acc | chunk: empty_chunk()}
  end

  defp finish_chunk(acc, timestamps?, tokenizer) do
    sequences = Enum.reverse(acc.previous_sequences)

    {token_ids, rest_token_ids} = merge_overlapping_sequences(sequences)

    # With timestamps we always finish chunks outside of context parts,
    # so we know the subsequent sequence is not going to overlap with
    # the chunk. Without timestamps we always need to keep the last
    # sequence for the next merge
    {chunk_token_ids, previous_sequences} =
      if timestamps? or rest_token_ids == [] do
        {token_ids ++ rest_token_ids, []}
      else
        {token_ids, [rest_token_ids]}
      end

    text = Bumblebee.Tokenizer.decode(tokenizer, chunk_token_ids)
    acc = put_in(acc.chunk.text, text)

    %{
      acc
      | previous_sequences: previous_sequences,
        chunks: [acc.chunk | acc.chunks],
        chunk: empty_chunk()
    }
  end

  defp empty_chunk(), do: %{start_timestamp_seconds: nil, end_timestamp_seconds: nil, text: nil}

  defp merge_overlapping_sequences([sequence]), do: {sequence, []}

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

    # We use binary backend so we are not blocked by the serving computation,
    # in this case we do simple operations with small data so it is fine
    sequences = Enum.map(sequences, &Nx.tensor(&1, backend: Nx.BinaryBackend))

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

    merged_sequence =
      Enum.reduce(acc, [], fn sequence, acc ->
        Nx.to_flat_list(sequence) ++ acc
      end)

    rest = Nx.to_flat_list(left_sequence)

    {merged_sequence, rest}
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
            {:ok, Nx.from_binary(data, :f32, backend: Nx.BinaryBackend)}

          {_, 1} ->
            {:error, "ffmpeg failed to decode the given file"}
        end
    end
  end
end
