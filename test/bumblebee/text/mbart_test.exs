defmodule Bumblebee.Text.MbartTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MBartModel"})

    assert %Bumblebee.Text.Mbart{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 16}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.8300, -0.4815, 0.4641], [-1.6583, 0.9162, -0.3562], [-0.6983, -0.7699, 1.0282]]
      ])
    )
  end

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-MBartForConditionalGeneration"}
             )

    assert %Bumblebee.Text.Mbart{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 250_027}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([[[0.0000, 0.0923, 0.0841], [0.0000, 0.1023, -0.0938], [0.0000, 0.0703, 0.1231]]])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-MBartForSequenceClassification"}
             )

    assert %Bumblebee.Text.Mbart{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 2, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[0.0085, 0.0054]])
    )
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-MBartForQuestionAnswering"}
             )

    assert %Bumblebee.Text.Mbart{architecture: :for_question_answering} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.start_logits) == {1, 10}
    assert Nx.shape(outputs.end_logits) == {1, 10}

    assert_all_close(
      outputs.start_logits[[.., 1..3]],
      Nx.tensor([[0.1063, -0.1271, -0.1534]])
    )

    assert_all_close(
      outputs.end_logits[[.., 1..3]],
      Nx.tensor([[0.0268, 0.0238, 0.0857]])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MBartForCausalLM"})

    assert %Bumblebee.Text.Mbart{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 250_027}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0000, -0.0236, -0.0043], [0.0000, -0.0101, 0.0510], [0.0000, 0.0404, 0.0327]]
      ])
    )
  end
end
