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

      assert Nx.Serving.run(serving, audio) == %{
               chunks: [
                 %{
                   text: " Tower of strength.",
                   start_timestamp_seconds: nil,
                   end_timestamp_seconds: nil
                 }
               ]
             }
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

      assert Nx.Serving.run(serving, audio) == %{
               chunks: [
                 %{
                   text:
                     " An awakening from the book of Irish poetry part 1, read for LibriVox.org by Sonja. An awakening by Alice Pirlong. O spring will wake in the heart of me with the rapture of blown violets, when the green bud quickens on every tree to spring will wake in the heart of me, and queues of honey",
                   start_timestamp_seconds: nil,
                   end_timestamp_seconds: nil
                 },
                 %{
                   text:
                     " will reign on the lee, tangling the grasses in silver nets. Yes, spring will awaken the heart of me with the rapture of blown violets. End of an awakening, this recording is in the public domain.",
                   start_timestamp_seconds: nil,
                   end_timestamp_seconds: nil
                 }
               ]
             }
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
          timestamps: :segments
        )

      audio =
        Path.join(@audio_dir, "librivox/46s_pcm_f32le_16000.bin")
        |> File.read!()
        |> Nx.from_binary(:f32)

      assert Nx.Serving.run(serving, audio) == %{
               chunks: [
                 %{
                   text:
                     " An awakening from the book of Irish poetry part 1, read for LibriVox.org by Sonia.",
                   start_timestamp_seconds: 0.0,
                   end_timestamp_seconds: 7.0
                 },
                 %{
                   text: " An awakening by Alice Pirlong.",
                   start_timestamp_seconds: 7.0,
                   end_timestamp_seconds: 11.0
                 },
                 %{
                   text:
                     " O spring will wake in the heart of me with the rapture of blown violets, when the green bud",
                   start_timestamp_seconds: 11.0,
                   end_timestamp_seconds: 18.12
                 },
                 %{
                   text:
                     " quickens on every tree to spring will wake in the heart of me, and queues of honey will reign on the lee,",
                   start_timestamp_seconds: 18.12,
                   end_timestamp_seconds: 25.92
                 },
                 %{
                   text:
                     " tangling the grasses in silver nets. Yes, spring will awaken the heart of me",
                   start_timestamp_seconds: 25.92,
                   end_timestamp_seconds: 32.48
                 },
                 %{
                   text: " with the rapture of blown violets.",
                   start_timestamp_seconds: 32.48,
                   end_timestamp_seconds: 34.88
                 },
                 %{
                   text: " End of an awakening, this recording is in the public domain.",
                   start_timestamp_seconds: 36.96,
                   end_timestamp_seconds: 40.72
                 }
               ]
             }
    end

    test "streaming without timestamps" do
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
          stream: true
        )

      audio =
        Path.join(@audio_dir, "librivox/46s_pcm_f32le_16000.bin")
        |> File.read!()
        |> Nx.from_binary(:f32)

      stream = Nx.Serving.run(serving, audio)

      assert Enum.to_list(stream) == [
               %{
                 text:
                   " An awakening from the book of Irish poetry part 1, read for LibriVox.org by Sonja. An awakening by Alice Pirlong. O spring will wake in the heart of me with the rapture of blown violets, when the green bud quickens on every tree to spring will wake in the heart of me, and queues of honey",
                 start_timestamp_seconds: nil,
                 end_timestamp_seconds: nil
               },
               %{
                 text:
                   " will reign on the lee, tangling the grasses in silver nets. Yes, spring will awaken the heart of me with the rapture of blown violets. End of an awakening, this recording is in the public domain.",
                 start_timestamp_seconds: nil,
                 end_timestamp_seconds: nil
               }
             ]
    end

    test "streaming with timestamps" do
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
          timestamps: :segments,
          stream: true
        )

      audio =
        Path.join(@audio_dir, "librivox/46s_pcm_f32le_16000.bin")
        |> File.read!()
        |> Nx.from_binary(:f32)

      stream = Nx.Serving.run(serving, audio)

      assert Enum.to_list(stream) == [
               %{
                 text:
                   " An awakening from the book of Irish poetry part 1, read for LibriVox.org by Sonia.",
                 start_timestamp_seconds: 0.0,
                 end_timestamp_seconds: 7.0
               },
               %{
                 text: " An awakening by Alice Pirlong.",
                 start_timestamp_seconds: 7.0,
                 end_timestamp_seconds: 11.0
               },
               %{
                 text:
                   " O spring will wake in the heart of me with the rapture of blown violets, when the green bud",
                 start_timestamp_seconds: 11.0,
                 end_timestamp_seconds: 18.12
               },
               %{
                 text:
                   " quickens on every tree to spring will wake in the heart of me, and queues of honey will reign on the lee,",
                 start_timestamp_seconds: 18.12,
                 end_timestamp_seconds: 25.92
               },
               %{
                 text:
                   " tangling the grasses in silver nets. Yes, spring will awaken the heart of me",
                 start_timestamp_seconds: 25.92,
                 end_timestamp_seconds: 32.48
               },
               %{
                 text: " with the rapture of blown violets.",
                 start_timestamp_seconds: 32.48,
                 end_timestamp_seconds: 34.88
               },
               %{
                 text: " End of an awakening, this recording is in the public domain.",
                 start_timestamp_seconds: 36.96,
                 end_timestamp_seconds: 40.72
               }
             ]
    end
  end
end
