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
        :client_batch_size,
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

    context_num_seconds =
      opts[:context_num_seconds] || (chunk_num_seconds && chunk_num_seconds / 6)

    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size])
        |> Shared.require_options!([:batch_size])
      end

    batch_size = compile[:batch_size]

    client_batch_size = opts[:client_batch_size] || batch_size || 1

    sampling_rate = featurizer.sampling_rate
    timestamps? = opts[:timestamps] != nil

    options = %{
      timestamps?: timestamps?,
      sampling_rate: sampling_rate,
      chunk_num_seconds: chunk_num_seconds,
      context_num_seconds: context_num_seconds
    }

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
      %{audio: audio, seed: seed} =
        case validate_input(input, sampling_rate, chunk_num_seconds) do
          {:ok, input} -> input
          {:error, message} -> raise ArgumentError, "invalid input, #{message}"
        end

      chunks =
        if chunk_num_seconds do
          chunk_input(audio, sampling_rate, chunk_num_seconds, context_num_seconds)
        else
          audio
        end

      stream =
        chunks
        |> Stream.chunk_every(client_batch_size)
        |> Stream.map(fn chunks ->
          seed =
            seed
            |> Nx.tensor(backend: Nx.BinaryBackend)
            |> Nx.broadcast({length(chunks)})

          inputs = Bumblebee.Featurizer.process_input(featurizer, chunks)
          Nx.Batch.concatenate([{inputs, seed}])
        end)

      {stream, {}}
    end)
    |> maybe_stream(opts[:stream], spec, featurizer, tokenizer, options)
  end

  defp validate_input(%{audio: audio} = input, sampling_rate, chunk_num_seconds) do
    with {:ok, audio} <- parse_audio(audio, sampling_rate, chunk_num_seconds) do
      {:ok, %{audio: audio, seed: input[:seed] || :erlang.system_time()}}
    end
  end

  defp validate_input(input, sampling_rate, chunk_num_seconds) do
    validate_input(%{audio: input}, sampling_rate, chunk_num_seconds)
  end

  defp parse_audio(input, sampling_rate, chunk_num_seconds) do
    case input do
      %Nx.Tensor{shape: {_}} = input ->
        {:ok, [Nx.backend_transfer(input, Nx.BinaryBackend)]}

      {:file, path} when is_binary(path) ->
        ffmpeg_read_as_pcm(path, sampling_rate)

      other ->
        cond do
          Enumerable.impl_for(other) == nil ->
            {:error,
             "expected audio to be a 1-dimensional tensor, an enumerable, or {:file, path}, got: #{inspect(other)}"}

          chunk_num_seconds == nil ->
            {:error,
             "enumerable input is only supported when chunking is enabled, make sure to set :chunk_num_seconds"}

          true ->
            stream =
              Stream.map(other, fn
                %Nx.Tensor{shape: {_}} = chunk ->
                  Nx.backend_transfer(chunk, Nx.BinaryBackend)

                item ->
                  raise ArgumentError,
                        "expected each enumerable item to be a 1-dimensional tensor, got: #{inspect(item)}"
              end)

            {:ok, stream}
        end
    end
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
        # This chunk can be of arbitrary size, the serving accumulates
        # and overlaps chunks internally as needed. We read the file
        # as stream to reduce memory usage
        chunk_size = 30

        stream =
          Stream.iterate(0, fn offset -> offset + chunk_size end)
          |> Stream.transform({}, fn offset, acc ->
            System.cmd(
              "ffmpeg",
              ~w[-ss #{offset} -t #{chunk_size} -i #{path} -ac #{channels} -ar #{sampling_rate} -f #{format} -hide_banner -loglevel quiet pipe:1]
            )
            |> case do
              {<<>>, 0} ->
                {:halt, acc}

              {data, 0} ->
                chunk = Nx.from_binary(data, :f32, backend: Nx.BinaryBackend)
                {[chunk], acc}

              {_, 1} ->
                raise "ffmpeg failed to decode the given file"
            end
          end)

        {:ok, stream}
    end
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

  # Takes a stream of continous tensor chunks and produces a stream
  # of overlapping chunks. As a result we get somewhat overlapping
  # transcriptions, which we merge at the edges to improve the final
  # transcription quality.
  defp chunk_input(stream, sampling_rate, chunk_num_seconds, context_num_seconds) do
    chunk_length = floor(chunk_num_seconds * sampling_rate)
    context_left = floor(context_num_seconds * sampling_rate)
    context_right = context_left

    step = chunk_length - context_left - context_right

    if step <= 0 do
      raise ArgumentError,
            ":chunk_num_seconds must be more than double the length of :context_num_seconds"
    end

    Stream.transform(
      stream,
      fn ->
        {[], 0}
      end,
      fn chunk, {buffer, buffer_size} ->
        buffer_size = buffer_size + Nx.size(chunk)
        buffer = buffer ++ [chunk]
        full_chunks([], {buffer, buffer_size}, chunk_length, step)
      end,
      fn {buffer, buffer_size} ->
        {[Nx.concatenate(buffer)], {buffer, buffer_size}}
      end,
      fn _ -> :ok end
    )
  end

  defp full_chunks(acc, {buffer, buffer_size}, chunk_length, _step)
       when chunk_length > buffer_size do
    {Enum.reverse(acc), {buffer, buffer_size}}
  end

  defp full_chunks(acc, {buffer, buffer_size}, chunk_length, step) do
    # We take `chunk_length` samples from the buffer, but drop only
    # `step`, since the rest we need for the next chunk
    {subchunks1, buffer} = slice_buffer(buffer, step, [])
    {subchunks2, _buffer} = slice_buffer(buffer, chunk_length - step, [])
    chunk = Nx.concatenate(subchunks1 ++ subchunks2)
    buffer_size = buffer_size - step
    full_chunks([chunk | acc], {buffer, buffer_size}, chunk_length, step)
  end

  defp slice_buffer(buffer, 0, acc), do: {Enum.reverse(acc), buffer}

  defp slice_buffer([chunk | buffer], size, acc) do
    chunk_size = Nx.size(chunk)

    if chunk_size <= size do
      slice_buffer(buffer, size - chunk_size, [chunk | acc])
    else
      {chunk, rest} = Nx.split(chunk, size)
      slice_buffer([rest | buffer], 0, [chunk | acc])
    end
  end

  defp maybe_stream(serving, false, spec, featurizer, tokenizer, options) do
    Nx.Serving.client_postprocessing(serving, fn {outputs, _metadata}, {} ->
      outputs = Nx.to_list(outputs)
      state = decode_chunk_outputs_init(spec, featurizer, tokenizer)
      {chunks, state} = decode_chunk_outputs_update(state, outputs, tokenizer, options)
      {final_chunks, _state} = decode_chunk_outputs_finish(state, tokenizer, options)
      chunks = chunks ++ final_chunks
      Shared.normalize_output([%{chunks: chunks}], false)
    end)
  end

  defp maybe_stream(serving, true, spec, featurizer, tokenizer, options) do
    serving
    |> Nx.Serving.streaming()
    |> Nx.Serving.client_postprocessing(fn stream, {} ->
      Stream.transform(
        stream,
        fn ->
          decode_chunk_outputs_init(spec, featurizer, tokenizer)
        end,
        fn {:batch, outputs, _metadata}, state ->
          outputs = Nx.to_list(outputs)
          decode_chunk_outputs_update(state, outputs, tokenizer, options)
        end,
        fn state ->
          decode_chunk_outputs_finish(state, tokenizer, options)
        end,
        fn _ -> :ok end
      )
    end)
  end

  # We generalize the decoding into multiple steps, where we feed a
  # number of outputs at a time. When not streaming we just feed all
  # outputs at once.

  defp decode_chunk_outputs_init(spec, featurizer, tokenizer) do
    time_precision = featurizer.num_seconds / spec.encoder_max_positions
    all_special_tokens = Bumblebee.Tokenizer.all_special_tokens(tokenizer)
    timestamp_begin_id = Bumblebee.Tokenizer.token_to_id(tokenizer, "<|notimestamps|>") + 1

    %{
      time_precision: time_precision,
      all_special_tokens: all_special_tokens,
      timestamp_begin_id: timestamp_begin_id,
      # Accumulation state
      time_offset: 0,
      right_context_end_time: nil,
      previous_sequences: [],
      current_sequence: [],
      chunks: [],
      chunk: empty_chunk()
    }
  end

  defp decode_chunk_outputs_update(state, outputs, tokenizer, options) do
    %{
      timestamps?: timestamps?,
      context_num_seconds: context_num_seconds
    } = options

    state =
      Enum.reduce(outputs, state, fn sequence, state ->
        process_output(state, sequence, tokenizer, options)
      end)

    state =
      cond do
        # If timestamps are enabled, chunks are finished on every end
        # timestamp, otherwise we need to do it explicitly
        timestamps? ->
          state

        # If context is enabled, we can emit chunks only once we have
        # two consecutive sequences, because we need to merge their
        # overlaps
        context_num_seconds == nil or context_num_seconds == 0 or
            at_least_2?(state.previous_sequences) ->
          finish_chunk(state, tokenizer, options)

        true ->
          state
      end

    pop_chunks(state)
  end

  defp decode_chunk_outputs_finish(state, tokenizer, options) do
    # Flush any pending sequences
    state = finish_chunk(state, tokenizer, options)
    pop_chunks(state)
  end

  defp at_least_2?([_, _ | _]), do: true
  defp at_least_2?(_), do: false

  defp pop_chunks(state) do
    get_and_update_in(state.chunks, fn chunks ->
      {Enum.reverse(chunks), []}
    end)
  end

  defp process_output(state, sequence, tokenizer, options) do
    %{
      timestamp_begin_id: timestamp_begin_id,
      time_precision: time_precision,
      all_special_tokens: all_special_tokens
    } = state

    %{
      sampling_rate: sampling_rate,
      chunk_num_seconds: chunk_num_seconds,
      context_num_seconds: context_num_seconds
    } = options

    chunk_length_seconds =
      chunk_num_seconds && floor(chunk_num_seconds * sampling_rate) / sampling_rate

    context_left_seconds =
      if(context_num_seconds,
        do: floor(context_num_seconds * sampling_rate) / sampling_rate,
        else: 0
      )

    context_right_seconds = context_left_seconds

    context_left_seconds = if(state.time_offset == 0, do: 0, else: context_left_seconds)

    time_offset = state.time_offset - context_left_seconds

    # We want to ignore timestamps in the right and left contexts,
    # because splitting on those would cause issues with merging
    # chunk overlaps. Also note that for right context, if there
    # are not timestamps in the right context, we ignore the last
    # regular timestamp, otherwise we may not have enough tokens
    # to merge chunks properly.
    #
    # Note that we don't know upfront whether the given chunk is the
    # last one (because we support input streaming), therefore we
    # always assume a non-zero right context (if enabled). However,
    # we keep track of the final end time in the right context in the
    # state (right_context_end_time), so once the input finishes
    # and we emit the last chunk, we can mark the correct end time.

    first_timestamp = timestamp_begin_id + context_left_seconds / time_precision

    {last_timestamp, right_context_last_timestamp} =
      if context_right_seconds > 0 do
        right_context_start = chunk_length_seconds - context_right_seconds

        sequence
        |> Enum.reverse()
        |> Enum.reduce_while({nil, nil}, fn token_id, {last_timestamp, right_context_end_time} ->
          if token_id >= timestamp_begin_id do
            right_context_end_time = right_context_end_time || token_id

            if last_timestamp != nil and
                 (token_id - timestamp_begin_id) * time_precision < right_context_start do
              {:halt, {last_timestamp, right_context_end_time}}
            else
              {:cont, {token_id, right_context_end_time}}
            end
          else
            {:cont, {last_timestamp, right_context_end_time}}
          end
        end)
      else
        {nil, nil}
      end

    right_context_end_time =
      if right_context_last_timestamp do
        time_offset + (right_context_last_timestamp - timestamp_begin_id) * time_precision
      end

    state =
      Enum.reduce(sequence, state, fn token_id, state ->
        if token_id >= timestamp_begin_id do
          time = (token_id - timestamp_begin_id) * time_precision + time_offset
          time = Float.round(time, 2)

          cond do
            last_timestamp && token_id >= last_timestamp ->
              state

            # If we are continuing previously open timestamp, ignore
            # timestamps within the left context
            state.previous_sequences != [] and token_id < first_timestamp ->
              state

            state.chunk.start_timestamp_seconds == nil ->
              put_in(state.chunk.start_timestamp_seconds, time)

            state.chunk.start_timestamp_seconds == time ->
              state

            true ->
              state = put_in(state.chunk.end_timestamp_seconds, time)

              state
              |> finish_current_sequence()
              |> finish_chunk(tokenizer, options)
          end
        else
          token = Bumblebee.Tokenizer.id_to_token(tokenizer, token_id)

          if token in all_special_tokens do
            # Skip special tokens
            state
          else
            %{state | current_sequence: [token_id | state.current_sequence]}
          end
        end
      end)

    state = finish_current_sequence(state)

    time_offset =
      if chunk_length_seconds do
        time_offset + chunk_length_seconds - context_right_seconds
      else
        time_offset
      end

    %{state | time_offset: time_offset, right_context_end_time: right_context_end_time}
  end

  defp finish_current_sequence(%{current_sequence: []} = state), do: state

  defp finish_current_sequence(state) do
    %{
      state
      | current_sequence: [],
        previous_sequences: [Enum.reverse(state.current_sequence) | state.previous_sequences]
    }
  end

  defp finish_chunk(%{previous_sequences: []} = state, _tokenizer, _options) do
    %{state | chunk: empty_chunk()}
  end

  defp finish_chunk(state, tokenizer, %{timestamps?: timestamps?}) do
    sequences = Enum.reverse(state.previous_sequences)

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

    state =
      update_in(state.chunk, fn chunk ->
        %{
          chunk
          | text: text,
            # We ignore timestamps in right chunk context, however once
            # we get to the last chunk, the right context is no longer
            # applicable, so we use the final end timestamp from the
            # last right context
            end_timestamp_seconds: chunk.end_timestamp_seconds || state.right_context_end_time
        }
      end)

    %{
      state
      | previous_sequences: previous_sequences,
        chunks: [state.chunk | state.chunks],
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
end
