defmodule Bumblebee.Text.MbartTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/mbart-large-cc25"},
                 architecture: :base
               )

      assert %Bumblebee.Text.Mbart{architecture: :base} = config

      input_ids = Nx.tensor([[35378, 4, 759, 10269, 83, 99942, 2, 250_004]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {1, 8, 1024}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-2.8804, -4.7890, -1.7658], [-3.0863, -4.9929, -1.2588], [-2.6020, -5.3808, -0.6461]]
        ]),
        atol: 1.0e-4
      )
    end

    test "conditional generation model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/mbart-large-en-ro"},
                 architecture: :for_conditional_generation,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_conditional_generation} = config

      input_ids = Nx.tensor([[4828, 83, 70, 35166, 2, 250_004]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 6, 250_027}

      assert_all_close(
        output.logits[[0, 1..3, 1..3]],
        Nx.tensor([
          [[3.6470, 11.0182, 3.5707], [3.5739, 7.6637, 1.8500], [3.2506, 8.7177, 2.7895]]
        ]),
        atol: 1.0e-4
      )
    end

    test "sequence classification model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-mbart"},
                 architecture: :for_sequence_classification,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_sequence_classification} = config

      input_ids = Nx.tensor([[157, 87, 21, 4, 44, 93, 43, 47, 70, 152, 16, 2, 1004]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[-0.0062, 0.0032]]),
        atol: 1.0e-4
      )
    end

    test "question answering model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-mbart"},
                 architecture: :for_question_answering,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_question_answering} = config

      input_ids = Nx.tensor([[8, 324, 53, 21, 22, 8, 338, 434, 157, 25, 7, 110, 153]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.start_logits) == {1, 13}
      assert Nx.shape(output.end_logits) == {1, 13}

      assert_all_close(
        output.start_logits[[0, 1..3]],
        Nx.tensor([-0.1411, 0.1579, 0.1181]),
        atol: 1.0e-4
      )

      assert_all_close(
        output.end_logits[[0, 1..3]],
        Nx.tensor([-0.0198, -0.2103, -0.1095]),
        atol: 1.0e-4
      )
    end

    test "causal language model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/mbart-large-cc25"},
                 architecture: :for_causal_language_modeling,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_causal_language_modeling} = config

      input_ids = Nx.tensor([[35378, 4, 759, 10269, 83, 99942, 2, 250_004]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 8, 250_027}

      assert_all_close(
        output.logits[[0, 1..3, 1..3]],
        Nx.tensor([
          [-0.1630, 20.1722, 20.1680],
          [-1.2354, 59.5818, 59.0031],
          [-2.2185, 94.7050, 92.3012]
        ]),
        atol: 1.0e-4
      )
    end
  end

  test "conditional generation" do
    {:ok, model, params, config} =
      Bumblebee.load_model({:hf, "facebook/mbart-large-en-ro"},
        architecture: :for_conditional_generation,
        module: Bumblebee.Text.Mbart
      )

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/mbart-large-en-ro"})

    assert %Bumblebee.Text.Mbart{architecture: :for_conditional_generation} = config

    english_phrase = "42 is the answer"

    inputs = Bumblebee.apply_tokenizer(tokenizer, english_phrase)

    token_ids =
      Bumblebee.Text.Generation.generate(config, model, params, inputs,
        min_length: 0,
        max_length: 6
      )

    assert Bumblebee.Tokenizer.decode(tokenizer, token_ids) == ["42 este răspunsul"]
  end
end
