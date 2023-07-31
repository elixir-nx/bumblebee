defmodule Bumblebee.Text.BlenderbotTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/blenderbot-400M-distill"},
                 architecture: :base
               )

      assert %Bumblebee.Text.Blenderbot{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[1710, 86, 1085, 2]]),
        "decoder_input_ids" => Nx.tensor([[1, 86]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 2, 1280}

      assert_all_close(
        outputs.hidden_state[[0, .., 1..3]],
        Nx.tensor([[0.1749, 0.4835, 0.3060], [0.0664, 0.0215, 0.5945]]),
        atol: 1.0e-4
      )
    end

    test "conditional generation model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "facebook/blenderbot-400M-distill"})

      assert %Bumblebee.Text.Blenderbot{architecture: :for_conditional_generation} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[1710, 86, 1085, 2]]),
        "decoder_input_ids" => Nx.tensor([[1, 86]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 2, 8008}

      assert_all_close(
        outputs.logits[[0, .., 1..3]],
        Nx.tensor([[12.0658, 3.7026, -4.7830], [-2.9581, 7.9437, -5.8420]]),
        atol: 1.0e-4
      )
    end
  end

  test "conditional generation" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/blenderbot-400M-distill"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "facebook/blenderbot-400M-distill"})

    assert %Bumblebee.Text.Blenderbot{architecture: :for_conditional_generation} = model_info.spec

    inputs = %{
      "input_ids" => Nx.tensor([[2675, 19, 544, 366, 304, 38, 2]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1]])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 5)

    generate =
      Bumblebee.Text.Generation.build_generate(
        model_info.model,
        model_info.spec,
        generation_config
      )

    token_ids = generate.(model_info.params, inputs)

    assert_equal(token_ids, Nx.tensor([[1, 281, 476, 929, 731, 2]]))
  end
end
