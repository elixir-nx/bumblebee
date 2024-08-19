defmodule Bumblebee.Text.NllbTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-nllb"},
               module: Bumblebee.Text.M2m100,
               architecture: :for_conditional_generation
             )

    assert %Bumblebee.Text.M2m100{architecture: :for_conditional_generation} = spec

    input = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    output = Axon.predict(model, params, input)

    assert Nx.shape(output.logits) == {1, 8, 128_112}

    assert_all_close(
      output.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [0.0000, 0.0169, -0.0698],
          [0.0000, 0.0525, -0.1042],
          [0.0000, 0.0667, -0.1078]
        ]
      ])
    )
  end
end
