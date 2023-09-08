defmodule Bumblebee.Audio do
  @moduledoc """
  High-level tasks related to audio processing.
  """

  # TODO: remove in v0.5
  @deprecated "Use Bumblebee.Audio.speech_to_text_whisper/5 instead."
  def speech_to_text(model_info, featurizer, tokenizer, generation_config, opts \\ []) do
    speech_to_text_whisper(model_info, featurizer, tokenizer, generation_config, opts)
  end

  @typedoc """
  A term representing audio.

  Can be either of:

    * a 1-dimensional `Nx.Tensor` with audio samples

    * `{:file, path}` with path to an audio file (note that this
      requires `ffmpeg` installed)

  """
  @type speech_to_text_whisper_input :: Nx.t() | {:file, String.t()}
  @type speech_to_text_whisper_output :: %{results: list(speech_to_text_whisper_result())}
  @type speech_to_text_whisper_result :: %{
          text: String.t(),
          chunks:
            list(%{
              text: String.t(),
              start_timestamp: number() | nil,
              end_timestamp: number() | nil
            })
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

    * `:timestamps` - when `true`, the model predicts timestamps for
      text segments (the length of each segment is up to the model)

    * `:seed` - random seed to use when sampling. By default the current
      timestamp is used

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
      on each of the devices. Defaults to `false`

  ## Examples

      {:ok, whisper} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai/whisper-tiny"})

      serving =
        Bumblebee.Audio.speech_to_text_whisper(whisper, featurizer, tokenizer, generation_config,
          defn_options: [compiler: EXLA]
        )

      Nx.Serving.run(serving, {:file, "/path/to/audio.wav"})
      #=> %{results: [%{text: "There is a cat outside the window."}]}

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
