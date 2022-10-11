defmodule Bumblebee.Text.BertTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "bert-base-uncased"}, architecture: :base)

      assert %Bumblebee.Text.Bert{architecture: :base} = spec

      input = %{
        "input_ids" => Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]]),
        "attention_mask" => Nx.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {1, 11, 768}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([[[0.4249, 0.1008, 0.7531], [0.3771, 0.1188, 0.7467], [0.4152, 0.1098, 0.7108]]]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, model, params, spec} = Bumblebee.load_model({:hf, "bert-base-uncased"})

      assert %Bumblebee.Text.Bert{architecture: :for_masked_language_modeling} = spec

      input = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 9, 30522}

      assert_all_close(
        output.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [-14.7240, -14.2120, -14.6434],
            [-10.3125, -9.7459, -9.9923],
            [-15.1105, -14.8048, -14.9276]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "sequence classification" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "textattack/bert-base-uncased-yelp-polarity"},
                 architecture: :for_sequence_classification
               )

      assert %Bumblebee.Text.Bert{architecture: :for_sequence_classification} = spec

      input = %{
        "input_ids" =>
          Nx.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 2002, 7317, 4747, 102]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[-1.3199, 1.5447]]),
        atol: 1.0e-4
      )
    end

    test "token classification" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "dbmdz/bert-large-cased-finetuned-conll03-english"})

      assert %Bumblebee.Text.Bert{architecture: :for_token_classification} = spec

      input = %{
        "input_ids" =>
          Nx.tensor([
            [101, 20164, 10932, 2271, 7954, 1110, 1359, 1107, 2123, 1105, 1203, 1365, 102]
          ])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 13, 9}

      assert_all_close(
        output.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-3.1215, -0.4028, -3.3213], [-2.4627, 0.0613, -3.2501], [-3.1475, -0.7705, -2.8248]]
        ]),
        atol: 1.0e-4
      )
    end

    test "question answering" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "deepset/bert-base-cased-squad2"})

      assert %Bumblebee.Text.Bert{architecture: :for_question_answering} = spec

      input = %{
        "input_ids" =>
          Nx.tensor([
            [101, 2627, 1108, 3104, 1124, 15703, 136] ++
              [102, 3104, 1124, 15703, 1108, 170, 3505, 16797, 102]
          ]),
        "token_type_ids" => Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.start_logits) == {1, 16}
      assert Nx.shape(output.end_logits) == {1, 16}

      assert_all_close(
        output.start_logits[[0..-1//1, 1..3]],
        Nx.tensor([[-6.9344, -6.9556, -2.8814]]),
        atol: 1.0e-4
      )

      assert_all_close(
        output.end_logits[[0..-1//1, 1..3]],
        Nx.tensor([[-7.3395, -7.9609, -7.4926]]),
        atol: 1.0e-4
      )
    end

    test "multiple choice" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "nightingal3/bert-finetuned-wsc"})

      assert %Bumblebee.Text.Bert{architecture: :for_multiple_choice} = spec

      input = %{
        "input_ids" =>
          Nx.tensor([
            [
              [101, 1999, 3304, 10733, 2003, 8828, 102, 2478, 9292, 1998, 5442, 1012, 102, 0],
              [101, 1999, 3304, 10733, 2003, 8828, 102, 2096, 2218, 1999, 1996, 2192, 1012, 102]
            ]
          ]),
        "attention_mask" =>
          Nx.tensor([
            [
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ]
          ]),
        "token_type_ids" =>
          Nx.tensor([
            [
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            ]
          ])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[0.3749, -3.9458]]),
        atol: 1.0e-4
      )
    end

    test "next sentence prediction" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "bert-base-uncased"},
                 architecture: :for_next_sentence_prediction
               )

      assert %Bumblebee.Text.Bert{architecture: :for_next_sentence_prediction} = spec

      input = %{
        "input_ids" =>
          Nx.tensor([
            [101, 1999, 3304, 10733, 2003, 2366, 4895, 14540, 6610, 2094, 1012] ++
              [102, 2059, 1996, 8013, 20323, 2009, 2478, 9292, 1998, 5442, 1012, 102]
          ]),
        "token_type_ids" =>
          Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[6.1459, -5.7820]]),
        atol: 1.0e-4
      )
    end

    test "causal language modeling" do
      assert {:ok, model, params, spec} =
               Bumblebee.load_model({:hf, "bert-base-uncased"},
                 architecture: :for_causal_language_modeling
               )

      assert %Bumblebee.Text.Bert{architecture: :for_causal_language_modeling} = spec

      input = %{
        "input_ids" => Nx.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 102]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 8, 30522}

      assert_all_close(
        output.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-6.0980, -6.1492, -6.0886], [-6.1857, -6.2198, -6.2982], [-6.3880, -6.3918, -6.3503]]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
