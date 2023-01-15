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

      actual = Bumblebee.apply_featurizer(featurizer, audio)[0]

      assert_all_close(actual, expected, atol: 1.0e-4, rtol: 1.0e-3)
    end
  end
end
