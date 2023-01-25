defmodule Bumblebee.Text.WhisperTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/whisper-tiny"}, architecture: :base)

      assert %Bumblebee.Audio.Whisper{architecture: :base} = spec

      input_features = Nx.sin(Nx.iota({1, 80, 3000}, type: :f32))
      decoder_input_ids = Nx.tensor([[50258, 50259, 50359, 50363]])

      inputs = %{
        "input_features" => input_features,
        "decoder_input_ids" => decoder_input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 4, 384}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 0..-1//1, 1..3]],
        Nx.tensor([
          [
            [5.0920, 0.6505, 1.4923],
            [-0.3086, -1.5719, -0.9174],
            [-1.0562, 2.0042, -9.0572],
            [-0.2586, -2.8140, -4.2827]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "for conditional generation model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/whisper-tiny"})

      assert %Bumblebee.Audio.Whisper{architecture: :for_conditional_generation} = spec

      input_features = Nx.sin(Nx.iota({1, 80, 3000}, type: :f32))
      decoder_input_ids = Nx.tensor([[50258, 50259, 50359, 50363]])

      inputs = %{
        "input_features" => input_features,
        "decoder_input_ids" => decoder_input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 4, 51865}

      assert_all_close(
        outputs.logits[[0..-1//1, 0..-1//1, 1..3]],
        Nx.tensor([
          [
            [-2.1714, 2.7831, 3.2897],
            [-8.5364, -5.1178, -6.9359],
            [19.6169, 16.3507, 16.4517],
            [-5.8487, -6.3570, -7.9041]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
