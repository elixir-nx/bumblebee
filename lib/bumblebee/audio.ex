defmodule Bumblebee.Audio do
  @moduledoc """
  High-level tasks related to audio processing.
  """

  @typedoc """
  A term representing audio.

  Can be either of:

    * a 1-dimensional `Nx.Tensor` with audio samples

    * `{:file, path}` with path to an audio file (note that this
      requires `ffmpeg` installed)

  """
  @type audio :: Nx.t() | {:file, String.t()}

  @type speech_to_text_whisper_input ::
          audio() | %{:audio => audio(), optional(:seed) => integer() | nil}
  @type speech_to_text_whisper_output :: %{
          chunks: list(speech_to_text_whisper_chunk())
        }
  @type speech_to_text_whisper_chunk :: %{
          text: String.t(),
          start_timestamp_seconds: number() | nil,
          end_timestamp_seconds: number() | nil
        }

  @doc """
  Builds serving for speech-to-text generation with Whisper models.

  The serving accepts `t:speech_to_text_whisper_input/0` and returns
  `t:speech_to_text_whisper_output/0`. A list of inputs is also supported.

  ## Options

    * `:chunk_num_seconds` - enables long-form transcription by splitting
      the input into chunks of the given length. Models generally have
      a limit on the input length, so by chunking we can feed smaller
      bits into the model, then merge the individual outputs into a
      single result at the end. By default chunking is disabled

    * `:context_num_seconds` - specifies the amount of overlap between
      chunks on both sides of split points. The context is effectively
      discarded when merging the chunks at the end, but it improves
      the results at the chunk edges. Note that the context is included
      in the total `:chunk_num_seconds`. Defaults to 1/6 of
      `:chunk_num_seconds`

    * `:language` - the language of the speech, when known upfront.
      Should be given as ISO alpha-2 code as string. By default no
      language is assumed and it is inferred from the input

    * `:task` - either of:

        * `:transcribe` (default) - generate audio transcription in
          the same language as the speech

        * `:translate` - generate translation of the given speech in
          English

    * `:timestamps` - when set, the model predicts timestamps and each
      annotated segment becomes an output chunk. Currently the only
      supported value is `:segments`, the length of each segment is up
      to the model

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

    * `:stream` - when `true`, the serving immediately returns a
      stream that emits chunks as they are generated. Note that
      when using streaming, only a single input can be given to the
      serving. To process a batch, call the serving with each input
      separately. Defaults to `false`

  ## Examples

      {:ok, whisper} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai/whisper-tiny"})

      serving =
        Bumblebee.Audio.speech_to_text_whisper(whisper, featurizer, tokenizer, generation_config,
          defn_options: [compiler: EXLA]
        )

      output = Nx.Serving.run(serving, {:file, "/path/to/audio.wav"})
      #=> %{
      #=>   chunks: [
      #=>     %{
      #=>       text: " There is a cat outside the window.",
      #=>       start_timestamp_seconds: nil,
      #=>       end_timestamp_seconds: nil
      #=>     }
      #=>   ]
      #=> }

      text = output.chunks |> Enum.map_join(& &1.text) |> String.trim()
      #=> "There is a cat outside the window."

  And with timestamps:

      serving =
        Bumblebee.Audio.speech_to_text_whisper(whisper, featurizer, tokenizer, generation_config,
          defn_options: [compiler: EXLA],
          chunk_num_seconds: 30,
          timestamps: :segments
        )

      Nx.Serving.run(serving, {:file, "/path/to/colouredstars_08_mathers_128kb.mp3"})
      #=> %{
      #=>   chunks: [
      #=>     %{
      #=>       text: " Such an eight of colored stars, versions of fifty isiatic love poems by Edward Powis-Mathers.",
      #=>       start_timestamp_seconds: 0.0,
      #=>       end_timestamp_seconds: 7.0
      #=>     },
      #=>     %{
      #=>       text: " This the revocs recording is in the public domain. Doubt. From the Japanese of Hori-Kawa,",
      #=>       start_timestamp_seconds: 7.0,
      #=>       end_timestamp_seconds: 14.0
      #=>     },
      #=>     %{
      #=>       text: " will he be true to me that I do not know. But since the dawn, I have had as much disorder in my thoughts as in my black hair, and of doubt.",
      #=>       start_timestamp_seconds: 14.0,
      #=>       end_timestamp_seconds: 27.0
      #=>     }
      #=>   ]
      #=> }

  """
  @spec speech_to_text_whisper(
          Bumblebee.model_info(),
          Bumblebee.Featurizer.t(),
          Bumblebee.Tokenizer.t(),
          Bumblebee.Text.GenerationConfig.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate speech_to_text_whisper(
                model_info,
                featurizer,
                tokenizer,
                generation_config,
                opts \\ []
              ),
              to: Bumblebee.Audio.SpeechToTextWhisper
end
