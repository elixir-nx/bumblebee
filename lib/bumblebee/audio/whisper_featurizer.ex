defmodule Bumblebee.Audio.WhisperFeaturizer do
  alias Bumblebee.Shared
  import Nx.Defn

  options = [
    feature_size: [
      default: 80,
      doc: "the feature dimension of the extracted features"
    ],
    sampling_rate: [
      default: 16000,
      doc: "the sampling rate at which the audio files should be digitally expressed in Hertz"
    ],
    hop_length: [
      default: 160,
      doc:
        "the length of the overlapping windows for the STFT used to obtain Mel Frequency coefficients"
    ],
    chunk_length: [
      default: 30,
      doc: """
      the maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter
      audio sequences
      """
    ],
    n_fft: [
      default: 400,
      doc: "size of the fourier transform"
    ],
    padding_value: [
      default: 0.0,
      doc: "padding value used to pad audio"
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
  def apply(featurizer, raw_samples) do
    raw_samples =
      for sample <- List.wrap(raw_samples) do
        if Nx.rank(sample) != 2 do
          Nx.new_axis(sample, -1)
        else
          sample
        end
      end

    max_length = featurizer.chunk_length * featurizer.sampling_rate

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
          extract_fbank_features(Nx.squeeze(waveform),
            nfft: featurizer.n_fft,
            sampling_rate: featurizer.sampling_rate,
            n_mels: featurizer.feature_size,
            hop_length: featurizer.hop_length
          )
        end)
        |> Nx.concatenate(axis: 0)
      end

    Nx.stack(transformed_samples)
  end

  defnp extract_fbank_features(waveform, opts \\ []) do
    opts = keyword!(opts, [:nfft, :sampling_rate, :n_mels, :hop_length])

    window = NxSignal.Windows.hann(n: opts[:nfft], is_periodic: true)

    {stft, _, _} =
      NxSignal.stft(waveform, window,
        fs: opts[:sampling_rate],
        nfft: opts[:nfft],
        overlap_size: opts[:nfft] - opts[:hop_length]
      )

    NxSignal.stft_to_mel(stft,
      fs: opts[:sampling_rate],
      nfft: opts[:nfft],
      n_mels: opts[:n_mels]
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
          chunk_length: {"chunk_length", number()},
          n_fft: {"n_fft", number()},
          padding_value: {"padding_value", number()}
        )

      @for.config(featurizer, opts)
    end
  end
end
