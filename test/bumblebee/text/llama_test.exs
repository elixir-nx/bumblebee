defmodule Bumblebee.Text.LlamaTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "seanmor5/tiny-llama-test"}, architecture: :base)

      assert %Bumblebee.Text.Llama{architecture: :base} = spec

      input_ids = Nx.tensor([[1, 15043, 3186, 825, 29915, 29879, 701]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 7, 32}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.4411, -1.9037, 0.9454], [0.8148, -1.4606, 0.0076], [0.9480, 0.6038, 0.1649]]
        ]),
        atol: 1.0e-4
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

    # test "causal language model" do
    #   assert {:ok, %{model: model, params: params, spec: spec}} =
    #            Bumblebee.load_model({:hf, "facebook/bart-base"},
    #              architecture: :for_causal_language_modeling
    #            )

    #   assert %Bumblebee.Text.Bart{architecture: :for_causal_language_modeling} = spec

    #   input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

    #   inputs = %{
    #     "input_ids" => input_ids
    #   }

    #   outputs = Axon.predict(model, params, inputs)

    #   assert Nx.shape(outputs.logits) == {1, 11, 50265}

    #   assert_all_close(
    #     outputs.logits[[0, 1..3, 1..3]],
    #     Nx.tensor([
    #       [-1.7658, -1.1057, -0.6313],
    #       [-1.0344, 4.4774, 0.5581],
    #       [-1.3625, 2.6272, -0.6478]
    #     ]),
    #     atol: 1.0e-4
    #   )
    # end
  end
end
