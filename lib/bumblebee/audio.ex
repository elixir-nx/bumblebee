defmodule Bumblebee.Audio do
  @moduledoc """
  High-level tasks related to audio processing.
  """

  @typedoc """
  A term representing audio.

  Can be either of:

    * a 2-dimensional `Nx.Tensor` with shape `{num_samples, num_channels}`

    * a path to an audio file (note this requires `ffmpeg` installed)

  """
  @type speech_recognition_input :: Nx.t() | String.t()
  @type speech_recognition_output :: %{results: list(speech_recognition_result())}
  @type speech_recognition_result :: %{text: String.t()}

  @doc """
  Builds serving for speech-to-text generation.

  The serving accepts `t:speech_recognition_input/0` and returns
  `t:speech_recognition_output/0`. A list of inputs is also supported.

  Note that either `:max_new_tokens` or `:max_length` must be specified.
  The generation should generally finish based on the audio input,
  however you still need to specify the upper limit.

  ## Options

    * `:max_new_tokens` - the maximum number of tokens to be generated,
      ignoring the number of tokens in the prompt

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  Also accepts all the other options of `Bumblebee.Text.Generation.build_generate/3`.

  ## Examples

      {:ok, whisper} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})

      serving =
        Bumblebee.Audio.speech_recognition(whisper, featurizer, tokenizer,
          max_new_tokens: 100,
          defn_options: [compiler: EXLA]
        )

      Nx.Serving.run(serving, "/path/to/audio.wav")
      #=> %{results: [%{text: "There is a cat outside the window."}]}

  """
  @spec speech_recognition(
          Bumblebee.model_info(),
          Bumblebee.Featurizer.t(),
          Bumblebee.Tokenizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate speech_recognition(model_info, featurizer, tokenizer, opts \\ []),
    to: Bumblebee.Audio.SpeechRecognition
end
