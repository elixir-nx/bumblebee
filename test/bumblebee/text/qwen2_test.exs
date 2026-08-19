defmodule Bumblebee.Text.Qwen2Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  defp inputs() do
    %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }
  end

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen2Model"})

    assert %Bumblebee.Text.Qwen2{architecture: :base} = spec
    assert spec.use_sliding_window == true
    assert spec.num_full_attention_blocks == 2

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.0997, 0.7288, 0.8140], [1.5144, -0.2792, 0.4321], [1.8058, 1.5185, 0.2998]]
      ])
    )

    # Positions beyond the sliding window of the upper blocks
    assert_all_close(
      outputs.hidden_state[[.., 5..7, 1..3]],
      Nx.tensor([
        [[-1.4559, 1.2817, -0.3535], [0.4296, -1.8397, 1.3706], [1.1167, -1.3683, -0.3627]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen2ForCausalLM"})

    assert %Bumblebee.Text.Qwen2{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.0867, 0.0547, 0.0222], [-0.1049, 0.0151, 0.1206], [-0.0252, 0.2101, -0.0344]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Qwen2ForSequenceClassification"}
             )

    assert %Bumblebee.Text.Qwen2{architecture: :for_sequence_classification} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(outputs.logits, Nx.tensor([[-0.0731, -0.0459]]))
  end
end
