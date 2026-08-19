defmodule Bumblebee.Text.KimiK25Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  # Kimi K2 uses the DeepSeek V3 architecture directly, while Kimi K2.5
  # is a multimodal model with a DeepSeek V3 text tower. In both cases
  # the text model is handled by `Bumblebee.Text.DeepseekV3`.

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Kimi_K25ForConditionalGeneration"}
             )

    assert %Bumblebee.Text.DeepseekV3{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1411, -0.0133, 0.0371], [0.0702, -0.0083, -0.0410], [0.2484, 0.1075, 0.0016]]
      ])
    )
  end
end
