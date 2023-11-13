defmodule Bumblebee.Text.MbartTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/mbart-large-cc25"},
                 architecture: :base
               )

      assert %Bumblebee.Text.Mbart{architecture: :base} = spec

      input_ids = Nx.tensor([[35_378, 4, 759, 10_269, 83, 99_942, 2, 250_004]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 8, 1024}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[-2.8804, -4.7890, -1.7658], [-3.0863, -4.9929, -1.2588], [-2.6020, -5.3808, -0.6461]]
        ]),
        atol: 1.0e-4
      )
    end

    test "conditional generation model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/mbart-large-en-ro"},
                 architecture: :for_conditional_generation,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_conditional_generation} = spec

      input_ids = Nx.tensor([[4828, 83, 70, 35_166, 2, 250_004]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 6, 250_027}

      assert_all_close(
        outputs.logits[[0, 1..3, 1..3]],
        Nx.tensor([
          [[3.6470, 11.0182, 3.5707], [3.5739, 7.6637, 1.8500], [3.2506, 8.7177, 2.7895]]
        ]),
        atol: 1.0e-4
      )
    end

    test "sequence classification model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-mbart"},
                 architecture: :for_sequence_classification,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_sequence_classification} = spec

      input_ids = Nx.tensor([[157, 87, 21, 4, 44, 93, 43, 47, 70, 152, 16, 2, 1004]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 2}

      assert_all_close(
        outputs.logits,
        Nx.tensor([[-0.0062, 0.0032]]),
        atol: 1.0e-4
      )
    end

    test "question answering model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-mbart"},
                 architecture: :for_question_answering,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_question_answering} = spec

      input_ids = Nx.tensor([[8, 324, 53, 21, 22, 8, 338, 434, 157, 25, 7, 110, 153]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.start_logits) == {1, 13}
      assert Nx.shape(outputs.end_logits) == {1, 13}

      assert_all_close(
        outputs.start_logits[[0, 1..3]],
        Nx.tensor([-0.1411, 0.1579, 0.1181]),
        atol: 1.0e-4
      )

      assert_all_close(
        outputs.end_logits[[0, 1..3]],
        Nx.tensor([-0.0198, -0.2103, -0.1095]),
        atol: 1.0e-4
      )
    end

    test "causal language model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/mbart-large-cc25"},
                 architecture: :for_causal_language_modeling,
                 module: Bumblebee.Text.Mbart
               )

      assert %Bumblebee.Text.Mbart{architecture: :for_causal_language_modeling} = spec

      input_ids = Nx.tensor([[35_378, 4, 759, 10_269, 83, 99_942, 2, 250_004]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 8, 250_027}

      assert_all_close(
        outputs.logits[[0, 1..3, 1..3]],
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
    {:ok, model_info} =
      Bumblebee.load_model({:hf, "facebook/mbart-large-en-ro"},
        architecture: :for_conditional_generation,
        module: Bumblebee.Text.Mbart
      )

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "facebook/mbart-large-en-ro"},
        spec_module: Bumblebee.Text.Mbart
      )

    assert %Bumblebee.Text.Mbart{architecture: :for_conditional_generation} = model_info.spec

    inputs = %{
      "input_ids" => Nx.tensor([[4828, 83, 70, 35_166, 2, 250_004]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1]])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 5)

    generate =
      Bumblebee.Text.Generation.build_generate(
        model_info.model,
        model_info.spec,
        generation_config
      )

    token_ids = generate.(model_info.params, inputs)

    assert_equal(token_ids, Nx.tensor([[250_020, 4828, 473, 54_051, 202, 2]]))
  end
end
