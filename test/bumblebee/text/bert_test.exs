defmodule Bumblebee.Text.BertTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BertModel"})

    assert %Bumblebee.Text.Bert{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.2331, 1.7817, 1.1736], [-1.1001, 1.3922, -0.3391], [0.0408, 0.8677, -0.0779]]
      ])
    )
  end

  test ":for_masked_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BertForMaskedLM"})

    assert %Bumblebee.Text.Bert{architecture: :for_masked_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1124}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.0127, 0.0508, 0.0904], [0.1151, 0.1189, 0.0922], [0.0089, 0.1132, -0.2470]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BertForSequenceClassification"}
             )

    assert %Bumblebee.Text.Bert{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0037, -0.0239]])
    )
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BertForTokenClassification"}
             )

    assert %Bumblebee.Text.Bert{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}

    assert_all_close(
      outputs.logits[[.., 1..3//1, ..]],
      Nx.tensor([[[0.2078, 0.0055], [0.0681, 0.1132], [0.1049, 0.0479]]])
    )
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BertForQuestionAnswering"}
             )

    assert %Bumblebee.Text.Bert{architecture: :for_question_answering} = spec

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
      Nx.tensor([[0.0465, 0.1204, 0.2137]])
    )

    assert_all_close(
      outputs.end_logits[[.., 1..3]],
      Nx.tensor([[0.1654, 0.0930, 0.1304]])
    )
  end

  test ":for_multiple_choice" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BertForMultipleChoice"})

    assert %Bumblebee.Text.Bert{architecture: :for_multiple_choice} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]]),
      "attention_mask" => Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]]),
      "token_type_ids" => Nx.tensor([[[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 1}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[0.0033]])
    )
  end

  test ":for_next_sentence_prediction" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BertForNextSentencePrediction"}
             )

    assert %Bumblebee.Text.Bert{architecture: :for_next_sentence_prediction} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "token_type_ids" => Nx.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0072, 0.0098]])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BertLMHeadModel"})

    assert %Bumblebee.Text.Bert{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1124}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.0498, 0.0272, -0.0722], [-0.2410, 0.1069, -0.2430], [-0.0683, 0.0077, -0.1277]]
      ])
    )
  end

  @tag :slow
  test ":for_colbert" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "colbert-ir/colbertv2.0"})

    assert %Bumblebee.Text.Bert{architecture: :for_colbert} = spec
    assert spec.embedding_size == 128

    inputs = %{
      "input_ids" => Nx.tensor([[101, 7592, 2088, 102]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.embeddings) == {1, 4, 128}

    # Values verified against Python transformers
    assert_all_close(
      outputs.embeddings[[.., 0..2, 0..4]],
      Nx.tensor([
        [
          [0.2045, 0.2541, -0.0945, 0.2049, 0.1241],
          [-0.0900, -0.2112, -0.00002, 1.0579, 0.8351],
          [0.3216, -0.0376, 0.0518, 0.0764, -0.4550]
        ]
      ]),
      atol: 1.0e-3
    )
  end
end
