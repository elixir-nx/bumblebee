defmodule Bumblebee.Text.ModernBertDecoderTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "onnx-internal-testing/tiny-random-ModernBertDecoderForCausalLM"},
               architecture: :base
             )

    assert %Bumblebee.Text.ModernBertDecoder{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, spec.hidden_size}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.5646, 0.0348, 2.0215], [-0.0864, -1.7016, 2.0514], [-1.2734, -0.2655, 1.8605]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "onnx-internal-testing/tiny-random-ModernBertDecoderForCausalLM"}
             )

    assert %Bumblebee.Text.ModernBertDecoder{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, spec.vocab_size}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[4.3767, 9.1061, -7.4934], [4.3417, -7.9699, -0.4800], [-1.0725, 9.2812, -2.1757]]
      ])
    )
  end
end
