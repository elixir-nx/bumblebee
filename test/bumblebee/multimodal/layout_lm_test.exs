defmodule Bumblebee.Multimodal.LayoutLmTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "microsoft/layoutlm-base-uncased"},
                 module: Bumblebee.Multimodal.LayoutLm,
                 architecture: :base
               )

      assert %Bumblebee.Multimodal.LayoutLm{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 7592, 2088, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1]]),
        "token_type_ids" => Nx.tensor([[0, 0, 0, 0]]),
        "bounding_box" =>
          Nx.tensor([
            [[0, 0, 0, 0], [637, 773, 693, 782], [698, 773, 733, 782], [1000, 1000, 1000, 1000]]
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 4, 768}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-0.0126, 0.2175, 0.1398], [0.0240, 0.5338, -0.1337], [-0.0190, 0.5194, 0.0706]]
        ]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "microsoft/layoutlm-base-uncased"},
                 module: Bumblebee.Multimodal.LayoutLm,
                 architecture: :for_masked_language_modeling
               )

      assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_masked_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[101, 7592, 2088, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1]]),
        "token_type_ids" => Nx.tensor([[0, 0, 0, 0]]),
        "bounding_box" =>
          Nx.tensor([
            [[0, 0, 0, 0], [637, 773, 693, 782], [698, 773, 733, 782], [1000, 1000, 1000, 1000]]
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 4, 30522}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-0.9018, -0.7695, 1.1371], [0.1485, -0.1378, 1.6499], [-0.5236, -0.4974, -0.6739]]
        ]),
        atol: 1.0e-4
      )
    end

    test "sequence classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "microsoft/layoutlm-base-uncased"},
                 module: Bumblebee.Multimodal.LayoutLm,
                 architecture: :for_sequence_classification
               )

      assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_sequence_classification} = spec

      params =
        update_in(params["sequence_classification_head.output"], fn %{"kernel" => k, "bias" => b} ->
          %{"kernel" => Nx.broadcast(1.0, k), "bias" => Nx.broadcast(0.0, b)}
        end)

      inputs = %{
        "input_ids" => Nx.tensor([[101, 7592, 2088, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1]]),
        "token_type_ids" => Nx.tensor([[0, 0, 0, 0]]),
        "bounding_box" =>
          Nx.tensor([
            [[0, 0, 0, 0], [637, 773, 693, 782], [698, 773, 733, 782], [1000, 1000, 1000, 1000]]
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 2}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[-0.6356, -0.6356]]),
        atol: 1.0e-4
      )
    end

    test "token classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "microsoft/layoutlm-base-uncased"},
                 module: Bumblebee.Multimodal.LayoutLm,
                 architecture: :for_token_classification
               )

      assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_token_classification} = spec

      params =
        update_in(params["token_classification_head.output"], fn %{"kernel" => k, "bias" => b} ->
          %{"kernel" => Nx.broadcast(1.0, k), "bias" => Nx.broadcast(0.0, b)}
        end)

      inputs = %{
        "input_ids" => Nx.tensor([[101, 7592, 2088, 102]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1]]),
        "token_type_ids" => Nx.tensor([[0, 0, 0, 0]]),
        "bounding_box" =>
          Nx.tensor([
            [[0, 0, 0, 0], [637, 773, 693, 782], [698, 773, 733, 782], [1000, 1000, 1000, 1000]]
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 4, 2}

      assert_all_close(
        outputs.logits,
        Nx.tensor([
          [[-9.0337, -9.0337], [-7.6490, -7.6490], [-6.9672, -6.9672], [-9.0373, -9.0373]]
        ]),
        atol: 1.0e-4
      )
    end

    test "question answering model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "impira/layoutlm-document-qa"},
                 module: Bumblebee.Multimodal.LayoutLm,
                 architecture: :for_question_answering
               )

      assert %Bumblebee.Multimodal.LayoutLm{architecture: :for_question_answering} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 20920, 232, 2]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1]]),
        "bounding_box" =>
          Nx.tensor([
            [[0, 0, 0, 0], [637, 773, 693, 782], [698, 773, 733, 782], [1000, 1000, 1000, 1000]]
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.start_logits) == {1, 4}
      assert Nx.shape(outputs.end_logits) == {1, 4}

      assert_all_close(
        outputs.start_logits,
        Nx.tensor([[-5.7846, -9.4211, -14.8011, -18.0101]]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.end_logits,
        Nx.tensor([[-7.8913, -11.3020, -10.6801, -19.1530]]),
        atol: 1.0e-4
      )
    end
  end
end
