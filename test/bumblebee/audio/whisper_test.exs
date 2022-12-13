defmodule Bumblebee.Text.WhisperTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/whisper-base"}, architecture: :base)

      assert %Bumblebee.Audio.Whisper{architecture: :base} = spec

      input_features = Nx.from_numpy("test/fixtures/inputs/whisper/test_input_features_1.npy")
      decoder_input_ids = Nx.tensor([[50258, 50258]])

      inputs = %{
        "input_features" => input_features,
        "decoder_input_ids" => decoder_input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 2, 512}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 0..-1//1, 1..3]],
        Nx.tensor([[[-2.0174, -1.4375, 2.8221], [-2.1639, -1.0226, 3.1215]]]),
        atol: 1.0e-4
      )
    end

    test "for conditional generation model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/whisper-tiny.en"})

      assert %Bumblebee.Audio.Whisper{architecture: :for_conditional_generation} = spec

      input_features = Nx.from_numpy("test/fixtures/inputs/whisper/test_input_features_2.npy")
      decoder_input_ids = Nx.tensor([[50257, 50257]])

      inputs = %{
        "input_features" => input_features,
        "decoder_input_ids" => decoder_input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 2, 51864}

      assert_all_close(
        outputs.logits[[0..-1//1, 0..-1//1, 1..3]],
        Nx.tensor([[[5.0992, 3.9357, 4.6614], [6.1992, 4.7241, 1.9518]]]),
        atol: 1.0e-4
      )
    end
  end
end
