defmodule Bumblebee.Audio.SpeechRecognitionTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @audio_dir Path.expand("../../fixtures/audio", __DIR__)

  describe "integration" do
    test "returns top scored labels" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})

      serving =
        Bumblebee.Audio.speech_recognition(model_info, featurizer, tokenizer,
          max_new_tokens: 100,
          defn_options: [compiler: EXLA]
        )

      audio =
        Path.join(
          @audio_dir,
          "common_voice/a6c7706a220eeea7ee3687c1122fe7ac17962d2449d25b6db37cc41cdaace442683e11945b6f581e73941c3083cd4eecfafc938840459cd8c571dae7774ee687_pcm_f32le_16000.bin"
        )
        |> File.read!()
        |> Nx.from_binary(:f32)
        |> Nx.reshape({:auto, 1})

      assert %{results: [%{text: "Tower of strength."}]} = Nx.Serving.run(serving, audio)
    end
  end
end
