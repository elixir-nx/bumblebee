defmodule Bumblebee.Audio.WhisperFeaturizer do
  alias Bumblebee.Shared

  import Nx.Defn

  options = [
    feature_size: [
      default: 80,
      doc: "the dimension of the extracted features. This corresponds to the number of Mel bins"
    ],
    sampling_rate: [
      default: 16_000,
      doc: "the sampling rate at which the audio files should be digitally expressed in Hertz"
    ],
    num_seconds: [
      default: 30,
      doc: """
      the maximum duration of the audio sequence. This implies that the the maximum length of the
      input sequence is `:num_seconds` * `:sampling_rate`
      """
    ],
    hop_length: [
      default: 160,
      doc:
        "the hop between consecutive overlapping windows for the STFT used to obtain Mel Frequency coefficients"
    ],
    fft_length: [
      default: 400,
      doc: "the size of the fourier transform"
    ],
    padding_value: [
      default: 0.0,
      doc: "the value used to pad the audio. Should correspond to silence"
    ]
  ]

  @moduledoc """
  Whisper featurizer for audio data.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Featurizer
  @behaviour Bumblebee.Configurable

  @impl true
  def config(featurizer, opts \\ []) do
    Shared.put_config_attrs(featurizer, opts)
  end

  @impl true
  def apply(featurizer, raw_samples, defn_options) do
    raw_samples =
      for sample <- List.wrap(raw_samples) do
        if Nx.rank(sample) != 2 do
          Nx.new_axis(sample, -1)
        else
          sample
        end
      end

    max_length = featurizer.num_seconds * featurizer.sampling_rate

    padded_samples =
      for sample <- raw_samples do
        pad_size = max_length - elem(Nx.shape(sample), 0)
        Nx.pad(sample, featurizer.padding_value, [{0, pad_size, 0}, {0, 0, 0}])
      end

    transformed_samples =
      for sample <- padded_samples do
        sample
        |> Nx.transpose()
        |> Nx.to_batched(1)
        |> Enum.map(fn waveform ->
          Nx.Defn.jit(&extract_fbank_features/2, defn_options).(Nx.squeeze(waveform),
            fft_length: featurizer.fft_length,
            sampling_rate: featurizer.sampling_rate,
            mel_bins: featurizer.feature_size,
            hop_length: featurizer.hop_length
          )
        end)
        |> Nx.concatenate(axis: 0)
      end

    samples = Nx.stack(transformed_samples)

    %{"input_features" => samples}
  end

  defnp extract_fbank_features(waveform, opts \\ []) do
    opts = keyword!(opts, [:fft_length, :sampling_rate, :mel_bins, :hop_length])

    window = NxSignal.Windows.hann(n: opts[:fft_length], is_periodic: true)

    {stft, _, _} =
      NxSignal.stft(waveform, window,
        sampling_rate: opts[:sampling_rate],
        fft_length: opts[:fft_length],
        overlap_length: opts[:fft_length] - opts[:hop_length],
        window_padding: :reflect
      )

    # Magic numbers taken from the reference implementation. This yields
    # max_mel ~ 3016
    frequency_spacing = 200.0 / 3
    max_mel = frequency_spacing * 45.245640471924965

    NxSignal.stft_to_mel(stft, opts[:sampling_rate],
      fft_length: opts[:fft_length],
      mel_bins: opts[:mel_bins],
      max_mel: max_mel,
      mel_frequency_spacing: frequency_spacing
    )
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(featurizer, data) do
      import Shared.Converters

      opts =
        convert!(data,
          feature_size: {"feature_size", number()},
          sampling_rate: {"sampling_rate", number()},
          hop_length: {"hop_length", number()},
          num_seconds: {"chunk_length", number()},
          fft_length: {"n_fft", number()},
          padding_value: {"padding_value", number()}
        )

      @for.config(featurizer, opts)
    end
  end
end
