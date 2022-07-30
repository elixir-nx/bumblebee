defmodule Bumblebee.Text.Gpt2Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "gpt2"}, architecture: :base)

      assert %Bumblebee.Text.Gpt2{architecture: :base} = config

      input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

      input = %{
        "input_ids" => input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {1, 11, 768}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [-0.0436, 0.0046, -0.1025],
            [-0.4822, 0.2564, 0.1926],
            [-0.2747, 0.0428, -0.1841]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "causal language modeling" do
      assert {:ok, model, params, config} = Bumblebee.load_model({:hf, "gpt2"})

      assert %Bumblebee.Text.Gpt2{architecture: :for_causal_language_modeling} = config

      input = %{
        "input_ids" => Nx.tensor([[15496, 11, 616, 3290, 318, 13779]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 6, 50257}

      assert_all_close(
        output.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [
            [-114.5832, -116.5725, -116.0830],
            [-89.8644, -93.1977, -94.4351],
            [-88.3380, -92.8703, -94.4454]
          ]
        ]),
        atol: 1.0e-4
      )
    end

    test "token classification" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "brad1141/gpt2-finetuned-comp2"})

      assert %Bumblebee.Text.Gpt2{architecture: :for_token_classification} = config

      input = %{
        "input_ids" => Nx.tensor([[15496, 11, 616, 3290, 318, 13779]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 6, 7}

      assert_all_close(
        output.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[0.4187, 3.4156, -2.8762], [2.9556, 0.9153, -1.0290], [1.3047, 1.0234, -1.2765]]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
