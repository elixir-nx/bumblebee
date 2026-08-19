defmodule Bumblebee.Text.Lfm2Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Lfm2Model"})

    assert %Bumblebee.Text.Lfm2{architecture: :base} = spec

    assert spec.block_types == [
             :conv,
             :conv,
             :full_attention,
             :conv,
             :conv,
             :full_attention
           ]

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.5004, 1.0087, 2.7270], [-0.9106, 2.0822, 0.7349], [0.3343, -0.1705, -0.1409]]
      ])
    )

    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[0.9100, 1.5279, 0.3757], [1.7871, 0.3324, 1.1375], [1.7054, -1.1016, -0.3343]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Lfm2ForCausalLM"})

    assert %Bumblebee.Text.Lfm2{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.1353, -0.1830, -0.1028], [0.0490, 0.0559, -0.1532], [-0.1779, 0.0222, -0.1727]]
      ])
    )
  end
end
