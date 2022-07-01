defmodule Bumblebee.Text.BartTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :slow
    @tag :capture_log
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/bart-base"}, architecture: :base)

      assert %Bumblebee.Text.Bart{architecture: :base} = config

      input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {1, 11, 768}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.3985, -1.2727, 1.8201], [1.2444, -1.5131, -0.9588], [-1.0806, -0.0743, 0.5012]]
        ]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "conditional generation model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/bart-base"},
                 architecture: :for_conditional_generation
               )

      assert %Bumblebee.Text.Bart{architecture: :for_conditional_generation} = config

      input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 11, 50265}

      assert_all_close(
        output.logits[[0, 1..3, 1..3]],
        Nx.tensor([
          [-4.3683, 2.3527, -4.6605],
          [-5.9831, 1.2762, -5.9307],
          [-5.8700, 5.1656, -6.0870]
        ]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    @tag timeout: 120_000
    test "sequence classification model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "valhalla/bart-large-sst2"})

      assert %Bumblebee.Text.Bart{architecture: :for_sequence_classification} = config
      input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[-0.1599, -0.0090]]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    @tag timeout: 120_000
    test "question answering model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "valhalla/bart-large-finetuned-squadv1"})

      assert %Bumblebee.Text.Bart{architecture: :for_question_answering} = config

      input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.start_logits) == {1, 11}
      assert Nx.shape(output.end_logits) == {1, 11}

      assert_all_close(
        output.start_logits[[0, 1..3]],
        Nx.tensor([-8.3735, -10.8867, -12.2982]),
        atol: 1.0e-4
      )

      assert_all_close(
        output.end_logits[[0, 1..3]],
        Nx.tensor([-8.7642, -7.8842, -11.4208]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "causal language model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/bart-base"},
                 architecture: :for_causal_language_modeling
               )

      assert %Bumblebee.Text.Bart{architecture: :for_causal_language_modeling} = config

      input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 11, 50265}

      assert_all_close(
        output.logits[[0, 1..3, 1..3]],
        Nx.tensor([
          [-1.7658, -1.1057, -0.6313],
          [-1.0344, 4.4774, 0.5581],
          [-1.3625, 2.6272, -0.6478]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
