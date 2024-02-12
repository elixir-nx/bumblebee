defmodule Bumblebee.Text.WhisperTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-WhisperModel"})

    assert %Bumblebee.Audio.Whisper{architecture: :base} = spec

    inputs = %{
      "input_features" => Nx.sin(Nx.iota({1, 60, 80}, type: :f32)),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 8, 16}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.3791, -1.6131, -0.6913], [0.1247, -1.3631, 0.0034], [-0.0097, 0.2039, 1.9897]]
      ])
    )
  end

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"}
             )

    assert %Bumblebee.Audio.Whisper{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_features" => Nx.sin(Nx.iota({1, 60, 80}, type: :f32)),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 50257}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0942, 0.1288, 0.0243], [-0.1667, -0.1401, 0.1191], [0.0398, -0.0449, -0.0574]]
      ])
    )
  end
end
