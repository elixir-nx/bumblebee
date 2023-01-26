defmodule Bumblebee.Text.AlbertTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "albert-base-v2"}, architecture: :base)

      assert %Bumblebee.Text.Albert{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]]),
        "attention_mask" => Nx.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 11, 768}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.6513, 1.5035, -0.2766], [-0.6515, 1.5046, -0.2780], [-0.6512, 1.5049, -0.2784]]
        ]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "albert-base-v2"})

      assert %Bumblebee.Text.Albert{architecture: :for_masked_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 9, 30000}

      assert_all_close(
        outputs.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[1.0450, -2.2835, -3.8152], [1.0635, -2.3124, -3.8890], [1.2576, -2.4207, -3.9500]]
        ]),
        atol: 1.0e-4
      )
    end

    test "sequence classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "textattack/albert-base-v2-imdb"})

      assert %Bumblebee.Text.Albert{architecture: :for_sequence_classification} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 2}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[0.4954, 0.1815]]),
        atol: 1.0e-4
      )
    end

    test "multiple choice model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "albert-base-v2"}, architecture: :for_multiple_choice)

      # The base is missing classifier params so we set them to
      # a static value here
      params =
        put_in(params["multiple_choice_head.output"], %{
          "kernel" => Nx.broadcast(1.0e-3, {spec.hidden_size, 1}),
          "bias" => Nx.tensor(0.0)
        })

      assert %Bumblebee.Text.Albert{architecture: :for_multiple_choice} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]]]),
        "attention_mask" => Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1]]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 1}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[-0.0200]]),
        atol: 1.0e-3
      )
    end

    test "token classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "vumichien/tiny-albert"})

      assert %Bumblebee.Text.Albert{architecture: :for_token_classification} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 9, 2}

      assert_all_close(
        outputs.logits[[0..-1//1, 1..3//1, 0..-1//1]],
        Nx.tensor([[[0.1364, -0.0437], [0.0360, -0.0786], [-0.1296, 0.0436]]]),
        atol: 1.0e-4
      )
    end

    test "question answering model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "twmkn9/albert-base-v2-squad2"})

      assert %Bumblebee.Text.Albert{architecture: :for_question_answering} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.start_logits) == {1, 9}
      assert Nx.shape(outputs.end_logits) == {1, 9}

      assert_all_close(
        outputs.start_logits[[0..-1//1, 1..3]],
        Nx.tensor([[-0.2464, -0.1028, -0.2076]]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.end_logits[[0..-1//1, 1..3]],
        Nx.tensor([[-1.3742, -1.3609, -1.3454]]),
        atol: 1.0e-4
      )
    end
  end
end
