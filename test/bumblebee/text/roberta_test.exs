defmodule Bumblebee.Text.RobertaTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-RobertaModel"})

    assert %Bumblebee.Text.Roberta{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([[[-0.3633, 0.8947, 1.8965], [0.5881, 1.9730, 1.4211], [0.8067, 1.6098, 0.0291]]])
    )
  end

  test ":for_masked_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-RobertaForMaskedLM"})

    assert %Bumblebee.Text.Roberta{architecture: :for_masked_language_modeling} = spec

    # TODO: remove once we load tied embeddings
    params = put_in(params["language_modeling_head.output"], params["embedder.token_embedding"])

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([[[0.0000, -0.0796, 0.1734], [0.0000, -0.0754, 0.0755], [0.0000, 0.0299, 0.1902]]])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-RobertaForSequenceClassification"}
             )

    assert %Bumblebee.Text.Roberta{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0032, 0.0017]])
    )
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-RobertaForTokenClassification"}
             )

    assert %Bumblebee.Text.Roberta{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}

    assert_all_close(
      outputs.logits[[.., 1..3//1, ..]],
      Nx.tensor([[[-0.0276, 0.0128], [-0.1321, 0.0960], [0.1680, 0.0699]]])
    )
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-RobertaForQuestionAnswering"}
             )

    assert %Bumblebee.Text.Roberta{architecture: :for_question_answering} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "token_type_ids" => Nx.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.start_logits) == {1, 10}
    assert Nx.shape(outputs.end_logits) == {1, 10}

    assert_all_close(
      outputs.start_logits[[.., 1..3]],
      Nx.tensor([[-0.1215, -0.1325, -0.1389]])
    )

    assert_all_close(
      outputs.end_logits[[.., 1..3]],
      Nx.tensor([[-0.2795, -0.0051, -0.1547]])
    )
  end

  test ":for_multiple_choice" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-RobertaForMultipleChoice"}
             )

    assert %Bumblebee.Text.Roberta{architecture: :for_multiple_choice} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]]),
      "attention_mask" => Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]]),
      "token_type_ids" => Nx.tensor([[[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 1}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0257]])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-RobertaForCausalLM"})

    assert %Bumblebee.Text.Roberta{architecture: :for_causal_language_modeling} = spec

    # TODO: remove once we load tied embeddings
    params = update_in(params, [Access.key!(:data)], fn data ->
      put_in(data["language_modeling_head.output"], data["embedder.token_embedding"])
    end)

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([[[0.0000, 0.0661, -0.0063], [0.0000, 0.1107, -0.1137], [0.0000, 0.1044, 0.0803]]])
    )
  end
end
