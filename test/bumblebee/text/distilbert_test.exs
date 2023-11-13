defmodule Bumblebee.Text.DistilbertTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "distilbert-base-uncased"}, architecture: :base)

      assert %Bumblebee.Text.Distilbert{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10_140, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 8, 768}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[0.1483, 0.3433, -0.5248], [0.5309, 0.3716, 0.0803], [0.3805, 0.5581, -0.4261]]
        ]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "distilbert-base-uncased"})

      assert %Bumblebee.Text.Distilbert{architecture: :for_masked_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 9, 30_522}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [
            [-14.1975, -13.9020, -13.9615],
            [-8.8192, -8.5549, -8.3866],
            [-13.4315, -13.2120, -13.3121]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "sequence classification" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "joeddav/distilbert-base-uncased-go-emotions-student"},
                 architecture: :for_sequence_classification
               )

      assert %Bumblebee.Text.Distilbert{architecture: :for_sequence_classification} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 1045, 2514, 5341, 2000, 2022, 2182, 1012, 102]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 28}

      assert_all_close(
        outputs.logits[[.., 1..4]],
        Nx.tensor([[-0.2951, -1.8836, -1.9071, 1.2820]]),
        atol: 1.0e-4
      )
    end

    test "token classification" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Davlan/distilbert-base-multilingual-cased-ner-hrl"})

      assert %Bumblebee.Text.Distilbert{architecture: :for_token_classification} = spec

      inputs = %{
        "input_ids" =>
          Nx.tensor([[101, 11_590, 11_324, 10_124, 14_290, 10_111, 146, 12_962, 10_106, 11_193, 102]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 11, 9}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-3.9901, -4.0522, -2.4171], [-4.0584, -4.2153, -2.4035], [-3.9693, -4.0597, -2.2356]]
        ]),
        atol: 1.0e-4
      )
    end

    test "question answering" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "distilbert-base-cased-distilled-squad"})

      assert %Bumblebee.Text.Distilbert{architecture: :for_question_answering} = spec

      inputs = %{
        "input_ids" =>
          Nx.tensor([
            [101, 2627, 1108, 3104, 1124, 15_703, 136, 102, 3104, 1124] ++
              [15_703, 1108, 170, 3505, 16_797, 102]
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.start_logits) == {1, 16}
      assert Nx.shape(outputs.end_logits) == {1, 16}

      assert_all_close(
        outputs.start_logits[[.., 1..3]],
        Nx.tensor([[-5.1663, -6.8352, -3.5082]]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.end_logits[[.., 1..3]],
        Nx.tensor([[-4.5860, -6.7391, -6.8987]]),
        atol: 1.0e-4
      )
    end
  end
end
