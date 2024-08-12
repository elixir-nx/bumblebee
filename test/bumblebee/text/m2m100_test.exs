defmodule Bumblebee.Text.M2m100Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-M2M100Model"},
               architecture: :base
             )

    assert %Bumblebee.Text.M2m100{architecture: :base} = spec

    input = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    output = Axon.predict(model, params, input)

    assert Nx.shape(output.hidden_state) == {1, 8, 16}

    assert_all_close(
      output.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [0.7856, -0.3174, -0.4792],
          [0.7265, -0.2752, -0.4823],
          [1.0580, -0.3263, -0.7994]
        ]
      ])
    )
  end

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-M2M100ForConditionalGeneration"},
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
          [0.0000, -0.0323, 0.0527],
          [0.0000, -0.0404, 0.0713],
          [0.0000, -0.0660, 0.0758]
        ]
      ])
    )
  end
end
