defmodule Bumblebee.Text.GptBigCodeTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPTBigCodeModel"})

      assert %Bumblebee.Text.GptBigCode{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[68, 69, 70, 266, 412, 8, 76, 396, 9, 26]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-0.8586, 0.3071, -0.3434], [-0.1530, 0.7143, -0.4393], [0.7845, 0.3625, -0.1734]]
        ]),
        atol: 1.0e-4
      )
    end

    test "base model without multi-query attention" do
      # We have a separate test to test parameter loading without
      # multi-query attention, because the parameters layout differs

      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model(
                 {:hf, "jonatanklosko/tiny-random-GPTBigCodeModel-multi_query-False"}
               )

      assert %Bumblebee.Text.GptBigCode{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[68, 69, 70, 266, 412, 8, 76, 396, 9, 26]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-1.3692, -0.4104, -1.2525], [-1.1314, 0.3077, -1.2131], [-0.5550, -0.0240, -1.1081]]
        ]),
        atol: 1.0e-4
      )
    end

    test "causal language modeling" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model(
                 {:hf, "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM"}
               )

      assert %Bumblebee.Text.GptBigCode{architecture: :for_causal_language_modeling} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[68, 69, 70, 266, 412, 8, 76, 396, 9, 26]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 10, 1024}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-0.0105, -0.0399, 0.1105], [-0.0350, 0.0781, 0.2945], [-0.1949, -0.1349, 0.0651]]
        ]),
        atol: 1.0e-4
      )
    end

    test "token classification" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model(
                 {:hf, "hf-internal-testing/tiny-random-GPTBigCodeForTokenClassification"}
               )

      assert %Bumblebee.Text.GptBigCode{architecture: :for_token_classification} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[68, 69, 70, 266, 412, 8, 76, 396, 9, 26]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 10, 2}

      assert_all_close(
        outputs.logits[[.., 1..3]],
        Nx.tensor([[[0.0179, -0.1119], [-0.1250, -0.0535], [-0.1324, 0.0488]]]),
        atol: 1.0e-4
      )
    end

    test "sequence classification" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model(
                 {:hf, "hf-internal-testing/tiny-random-GPTBigCodeForSequenceClassification"}
               )

      assert %Bumblebee.Text.GptBigCode{architecture: :for_sequence_classification} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[68, 69, 70, 266, 412, 8, 76, 396, 9, 26]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 2}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[0.1027, 0.2042]]),
        atol: 1.0e-4
      )
    end
  end
end
