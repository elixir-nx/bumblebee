defmodule Bumblebee.Text.MuseGlimmerTextTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-MuseGlimmerTextModel"})

    assert %Bumblebee.Text.MuseGlimmerText{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.0428, 1.3647, -0.1806], [0.5728, 1.5825, -0.4060], [1.3033, 1.4808, -0.4369]]
      ])
    )

    # Positions beyond the sliding window of the local attention blocks
    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[0.4745, 1.9581, -0.2626], [-1.4397, 1.0010, -0.0487], [-0.3937, 0.2371, -0.2116]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    # The language modeling head is only a part of the multimodal model,
    # so we load the text tower out of a multimodal checkpoint
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-MuseGlimmerForConditionalGeneration"}
             )

    assert %Bumblebee.Text.MuseGlimmerText{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0181, -0.0014, 0.0203], [0.0034, -0.0273, -0.0016], [0.0654, 0.0345, -0.0055]]
      ])
    )
  end
end
