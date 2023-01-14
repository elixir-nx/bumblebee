defmodule Bumblebee.Audio.WhisperFeaturizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  require IEx

  describe "integration" do
    test "encoding model input" do
      assert {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny.en"})

      assert %Bumblebee.Audio.WhisperFeaturizer{} = featurizer

      audio = Nx.from_numpy("test/fixtures/inputs/whisper/test_input_raw_1.npy")
      expected = Nx.from_numpy("test/fixtures/inputs/whisper/test_output_featurized_1.npy")

      actual = Bumblebee.apply_featurizer(featurizer, audio)

      IEx.pry()

      assert_all_close(actual[[0, 0..-1//1, 0..-2//1]][0], expected)
    end
  end
end
