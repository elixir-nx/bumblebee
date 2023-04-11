defmodule Bumblebee.Text.WhisperTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/whisper-tiny"}, architecture: :base)

      assert %Bumblebee.Audio.Whisper{architecture: :base} = spec

      input_features = Nx.sin(Nx.iota({1, 3000, 80}, type: :f32))
      decoder_input_ids = Nx.tensor([[50258, 50259, 50359, 50363]])

      inputs = %{
        "input_features" => input_features,
        "decoder_input_ids" => decoder_input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 4, 384}

      assert_all_close(
        outputs.hidden_state[[.., .., 1..3]],
        Nx.tensor([
          [
            [9.1349, 0.5695, 8.7758],
            [0.0160, -7.0785, 1.1313],
            [6.1074, -2.0481, -1.5687],
            [5.6247, -10.3924, 7.2008]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "for conditional generation model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "openai/whisper-tiny"})

      assert %Bumblebee.Audio.Whisper{architecture: :for_conditional_generation} = spec

      input_features = Nx.sin(Nx.iota({1, 3000, 80}, type: :f32))
      decoder_input_ids = Nx.tensor([[50258, 50259, 50359, 50363]])

      inputs = %{
        "input_features" => input_features,
        "decoder_input_ids" => decoder_input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 4, 51865}

      assert_all_close(
        outputs.logits[[.., .., 1..3]],
        Nx.tensor([
          [
            [2.0805, 6.0644, 7.0570],
            [-7.8065, -3.0313, -5.1049],
            [17.4098, 16.2510, 16.0446],
            [-7.7142, -5.9466, -6.1812]
          ]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
