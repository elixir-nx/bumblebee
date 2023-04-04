defmodule Bumblebee.Text.RobertaTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "roberta-base"}, architecture: :base)

      assert %Bumblebee.Text.Roberta{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 11, 768}

      assert_all_close(
        outputs.hidden_state[[.., 0..2, 0..2]],
        Nx.tensor([
          [[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]
        ]),
        atol: 1.0e-4
      )
    end

    test "masked language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "roberta-base"})

      assert %Bumblebee.Text.Roberta{architecture: :for_masked_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 11, 50265}

      assert_all_close(
        outputs.logits[[.., 0..2, 0..2]],
        Nx.tensor([
          [[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]
        ]),
        atol: 1.0e-4
      )
    end

    test "sequence classification" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "cardiffnlp/twitter-roberta-base-emotion"})

      assert %Bumblebee.Text.Roberta{architecture: :for_sequence_classification} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 31414, 6, 127, 2335, 16, 11962, 37, 11639, 1168, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 4}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[-1.3661, 3.0174, -0.9609, -0.4145]]),
        atol: 1.0e-4
      )
    end

    test "token classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Jean-Baptiste/roberta-large-ner-english"})

      assert %Bumblebee.Text.Roberta{architecture: :for_token_classification} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[30581, 3923, 34892, 16, 10, 138, 716, 11, 2201, 8, 188, 469]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 12, 5}

      assert_all_close(
        outputs.logits[[.., 0..2, 0..1]],
        Nx.tensor([[[4.1969, -2.5614], [-1.4174, -0.6959], [-1.3807, 0.1313]]]),
        atol: 1.0e-4
      )
    end

    test "question answering model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"})

      assert %Bumblebee.Text.Roberta{architecture: :for_question_answering} = spec

      inputs = %{
        "input_ids" =>
          Nx.tensor([
            [0, 12375, 21, 2488, 289, 13919, 116, 2, 2, 24021, 289, 13919, 21, 10, 2579, 29771, 2]
          ])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.start_logits) == {1, 17}
      assert Nx.shape(outputs.end_logits) == {1, 17}

      assert_all_close(
        outputs.start_logits[[.., 0..2]],
        Nx.tensor([[0.5901, -8.3490, -8.8031]]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.end_logits[[.., 0..2]],
        Nx.tensor([[1.1207, -7.5968, -7.6151]]),
        atol: 1.0e-4
      )
    end

    test "multiple choice model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "LIAMF-USP/aristo-roberta"})

      assert %Bumblebee.Text.Roberta{architecture: :for_multiple_choice} = spec

      inputs = %{
        "input_ids" =>
          Nx.tensor([
            [[0, 38576, 103, 4437, 2, 2, 725, 895, 2], [0, 38576, 103, 4437, 2, 2, 487, 895, 2]]
          ]),
        "attention_mask" =>
          Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 2}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[-13.9123, -13.4582]]),
        atol: 1.0e-3
      )
    end

    test "casual language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "roberta-base"},
                 architecture: :for_causal_language_modeling
               )

      assert %Bumblebee.Text.Roberta{architecture: :for_causal_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[0, 31414, 6, 127, 2335, 16, 11962, 2]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 8, 50265}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-3.3435, 32.1472, -3.5083], [-3.5373, 21.8191, -3.5197], [-4.2189, 22.5419, -3.9859]]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
