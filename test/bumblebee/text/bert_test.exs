defmodule Bumblebee.Text.BertTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "bert-base-uncased"}, architecture: :base)

      assert %Bumblebee.Text.Bert{architecture: :base} = config

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

    test "sequence classification" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "textattack/bert-base-uncased-yelp-polarity"},
                 architecture: :for_sequence_classification
               )

      assert %Bumblebee.Text.Bert{architecture: :for_sequence_classification} = config

      input = %{
        "input_ids" =>
          Nx.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 2002, 7317, 4747, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[-1.3199, 1.5447]]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, model, params, config} = Bumblebee.load_model({:hf, "bert-base-uncased"})

      assert %Bumblebee.Text.Bert{architecture: :for_masked_language_modeling} = config

      input = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
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
  end
end
