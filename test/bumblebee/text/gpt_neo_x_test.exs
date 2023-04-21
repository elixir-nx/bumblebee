defmodule Bumblebee.Text.GptNeoXTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "seanmor5/tiny-gpt-neox-test"}, architecture: :base)

      assert %Bumblebee.Text.GptNeoX{architecture: :base} = spec

      input_ids = Nx.tensor([[4, 928, 219, 10, 591, 1023]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 6, 32}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[1.4331, 0.7042, -2.8534], [1.4009, 1.5367, -0.8567], [0.7013, 1.5902, -1.4052]]
        ]),
        atol: 1.0e-2
      )
    end

    # test "sequence classification model" do
    #   assert {:ok, %{model: model, params: params, spec: spec}} =
    #            Bumblebee.load_model({:hf, "valhalla/bart-large-sst2"})

    #   assert %Bumblebee.Text.Bart{architecture: :for_sequence_classification} = spec
    #   input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

    #   inputs = %{
    #     "input_ids" => input_ids
    #   }

    #   outputs = Axon.predict(model, params, inputs)

    #   assert Nx.shape(outputs.logits) == {1, 2}

    #   assert_all_close(
    #     outputs.logits,
    #     Nx.tensor([[-0.1599, -0.0090]]),
    #     atol: 1.0e-4
    #   )
    # end

    test "causal language model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "seanmor5/tiny-gpt-neox-test"},
                 architecture: :for_causal_language_modeling
               )

      assert %Bumblebee.Text.GptNeoX{architecture: :for_causal_language_modeling} = spec

      input_ids = Nx.tensor([[4, 928, 219, 10, 591, 1023]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 6, 1024}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [[0.0559, 0.1583, 0.0423], [0.0446, 0.0843, -0.0328], [0.1069, 0.0430, 0.0127]]
        ]),
        atol: 1.0e-2
      )
    end
  end
end
