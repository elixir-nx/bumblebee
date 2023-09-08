defmodule Bumblebee.Audio.SpeechToTextWhisperTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @audio_dir Path.expand("../../fixtures/audio", __DIR__)

  describe "integration" do
    test "generates transcription" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai/whisper-tiny"})

      serving =
        Bumblebee.Audio.speech_to_text_whisper(
          model_info,
          featurizer,
          tokenizer,
          generation_config,
          defn_options: [compiler: EXLA]
        )

      audio =
        Path.join(
          @audio_dir,
          "common_voice/a6c7706a220eeea7ee3687c1122fe7ac17962d2449d25b6db37cc41cdaace442683e11945b6f581e73941c3083cd4eecfafc938840459cd8c571dae7774ee687_pcm_f32le_16000.bin"
        )
        |> File.read!()
        |> Nx.from_binary(:f32)

      assert %{results: [%{text: "Tower of strength."}]} = Nx.Serving.run(serving, audio)
    end

    test "long-form transcription with chunking" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai/whisper-tiny"})

      serving =
        Bumblebee.Audio.speech_to_text_whisper(
          model_info,
          featurizer,
          tokenizer,
          generation_config,
          chunk_num_seconds: 30,
          defn_options: [compiler: EXLA]
        )

      audio =
        Path.join(@audio_dir, "librivox/46s_pcm_f32le_16000.bin")
        |> File.read!()
        |> Nx.from_binary(:f32)

      transcription =
        "An awakening from the book of Irish poetry part 1, read for LibriVox.org by Sonia. An awakening by Alice Pirlong. O spring will wake in the heart of me with the rapture of blown violets, when the green bud quickens on every tree to spring will wake in the heart of me, and queues of honey will reign on the lee, tangling the grasses in silver nets. Yes, spring will awaken the heart of me with the rapture of blown violets. End of an awakening, this recording is in the public domain."

      assert %{results: [%{text: ^transcription}]} = Nx.Serving.run(serving, audio)
    end

    test "long-form transcription with timestamps" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai/whisper-tiny"})

      serving =
        Bumblebee.Audio.speech_to_text_whisper(
          model_info,
          featurizer,
          tokenizer,
          generation_config,
          chunk_num_seconds: 30,
          defn_options: [compiler: EXLA],
          timestamps: true
        )

      audio =
        Path.join(@audio_dir, "librivox/46s_pcm_f32le_16000.bin")
        |> File.read!()
        |> Nx.from_binary(:f32)

      transcription =
        "An awakening from the book of Irish poetry part 1, read for LibriVox.org by Sonia. An awakening by Alice Pirlong. O spring will wake in the heart of me with the rapture of blown violets, when the green bud quickens on every tree to spring will wake in the heart of me, and queues of honey will reign on the lee, tangling the grasses in silver nets. Yes, spring will awaken the heart of me with the rapture of blown violets. End of an awakening, this recording is in the public domain."

      assert %{results: [%{text: ^transcription, chunks: chunks}]} =
               Nx.Serving.run(serving, audio)

      assert chunks == [
               %{
                 text:
                   " An awakening from the book of Irish poetry part 1, read for LibriVox.org by Sonia.",
                 start_timestamp: 0.0,
                 end_timestamp: 7.0
               },
               %{
                 text: " An awakening by Alice Pirlong.",
                 start_timestamp: 7.0,
                 end_timestamp: 11.0
               },
               %{
                 text:
                   " O spring will wake in the heart of me with the rapture of blown violets, when the green bud",
                 start_timestamp: 11.0,
                 end_timestamp: 18.12
               },
               %{
                 text:
                   " quickens on every tree to spring will wake in the heart of me, and queues of honey will reign on the lee,",
                 start_timestamp: 18.12,
                 end_timestamp: 25.92
               },
               %{
                 text:
                   " tangling the grasses in silver nets. Yes, spring will awaken the heart of me",
                 start_timestamp: 25.92,
                 end_timestamp: 32.48
               },
               %{
                 text: " with the rapture of blown violets.",
                 start_timestamp: 32.48,
                 end_timestamp: 34.88
               },
               %{
                 text: " End of an awakening, this recording is in the public domain.",
                 start_timestamp: 36.96,
                 end_timestamp: 40.72
               }
             ]
    end
  end
end
