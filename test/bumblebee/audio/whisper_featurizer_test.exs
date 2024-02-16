defmodule Bumblebee.Audio.WhisperFeaturizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encodes text" do
      assert {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})

      assert %Bumblebee.Audio.WhisperFeaturizer{} = featurizer

      audio = Nx.sin(Nx.iota({100}, type: :f32))

      inputs = Bumblebee.apply_featurizer(featurizer, audio, defn_options: [compiler: EXLA])

      assert_all_close(
        inputs["input_features"][[0, 0..3, 0..3]],
        Nx.tensor([
          [
            [0.7313, 0.7820, 0.7391, 0.6787],
            [0.4332, 0.4861, 0.4412, 0.3497],
            [-0.5938, -0.5938, -0.5938, -0.5938],
            [-0.5938, -0.5938, -0.5938, -0.5938]
          ]
        ])
      )
    end
  end
end
