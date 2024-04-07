defmodule Bumblebee.Text.BlenderbotTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BlenderbotModel"})

    assert %Bumblebee.Text.Blenderbot{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 8, 16}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.6578, 1.9730, 0.6908], [-1.8067, 0.0553, -0.7491], [0.1820, -0.4390, -0.8273]]
      ])
    )
  end

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BlenderbotForConditionalGeneration"}
             )

    assert %Bumblebee.Text.Blenderbot{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0440, -0.0115, -0.0004], [0.0772, 0.0327, -0.0667], [-0.0419, 0.1483, 0.0140]]
      ])
    )
  end
end
