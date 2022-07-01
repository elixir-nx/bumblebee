defmodule Bumblebee.Text.AlbertTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :slow
    @tag :capture_log
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "albert-base-v2"}, architecture: :base)

      assert %Bumblebee.Text.Albert{architecture: :base} = config

      input = %{
        "input_ids" => Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]]),
        "attention_mask" => Nx.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {1, 11, 768}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.6513, 1.5035, -0.2766], [-0.6515, 1.5046, -0.2780], [-0.6512, 1.5049, -0.2784]]
        ]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "masked language modeling model" do
      assert {:ok, model, params, config} = Bumblebee.load_model({:hf, "albert-base-v2"})

      assert %Bumblebee.Text.Albert{architecture: :for_masked_language_modeling} = config

      input = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 9, 30000}

      assert_all_close(
        output.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[1.0450, -2.2835, -3.8152], [1.0635, -2.3124, -3.8890], [1.2576, -2.4207, -3.9500]]
        ]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "sequence classification model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "textattack/albert-base-v2-imdb"})

      assert %Bumblebee.Text.Albert{architecture: :for_sequence_classification} = config

      input = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[0.4954, 0.1815]]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "multiple choice model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "albert-base-v2"}, architecture: :for_multiple_choice)

      # The base is missing classifier params so we set
      # them to a static val here
      classifier_params = %{
        "kernel" => Nx.broadcast(1.0e-3, {config.hidden_size, 1}),
        "bias" => Nx.tensor(0.0)
      }

      params = Map.replace(params, "classifier", classifier_params)

      assert %Bumblebee.Text.Albert{architecture: :for_multiple_choice} = config

      input = %{
        "input_ids" => Nx.tensor([[[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]]]),
        "attention_mask" => Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1]]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 1}

      assert_all_close(
        output.logits,
        Nx.tensor([[-0.0200]]),
        atol: 1.0e-3
      )
    end

    @tag :slow
    @tag :capture_log
    test "token classification model" do
      assert {:ok, model, params, config} = Bumblebee.load_model({:hf, "vumichien/tiny-albert"})

      assert %Bumblebee.Text.Albert{architecture: :for_token_classification} = config

      input = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 9, 2}

      assert_all_close(
        output.logits[[0..-1//1, 1..3//1, 0..-1//1]],
        Nx.tensor([[[0.1364, -0.0437], [0.0360, -0.0786], [-0.1296, 0.0436]]]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "question answering model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "twmkn9/albert-base-v2-squad2"})

      assert %Bumblebee.Text.Albert{architecture: :for_question_answering} = config

      input = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.start_logits) == {1, 9}
      assert Nx.shape(output.end_logits) == {1, 9}

      assert_all_close(
        output.start_logits[[0..-1//1, 1..3]],
        Nx.tensor([[-0.2464, -0.1028, -0.2076]]),
        atol: 1.0e-4
      )

      assert_all_close(
        output.end_logits[[0..-1//1, 1..3]],
        Nx.tensor([[-1.3742, -1.3609, -1.3454]]),
        atol: 1.0e-4
      )
    end
  end
end
