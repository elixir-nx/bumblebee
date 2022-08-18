defmodule Bumblebee.Text.M2m100Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/m2m100_418M"},
                 architecture: :base
               )

      assert %Bumblebee.Text.M2m100{architecture: :base} = config

      # This model always accepts decoder_input_ids which are the current
      # sequence with padding, in this case we give english source sequence
      # and french target token with padding
      input_ids = Nx.tensor([[128_022, 21457, 117, 14906, 8, 37089, 432, 110_309, 10550, 2]])
      attention_mask = Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      decoder_input_ids = Nx.tensor([[128_028, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

      input = %{
        "input_ids" => input_ids,
        attention_mask: attention_mask,
        decoder_input_ids: decoder_input_ids
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {1, 10, 1024}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-2.1096, 0.5567, 0.0736], [-2.2717, 0.5756, 0.0586], [-2.3519, 0.4933, 0.0666]]
        ]),
        atol: 1.0e-4
      )
    end

    # test "for conditional generation model" do
    #   assert {:ok, model, params, config} =
    #            Bumblebee.load_model({:hf, "facebook/m2m100_418M"},
    #              architecture: :for_conditional_generation
    #            )

    #   assert %Bumblebee.Text.M2m100{architecture: :for_conditional_generation} = config

    #   # This model always accepts decoder_input_ids which are the current
    #   # sequence with padding, in this case we give english source sequence
    #   # and french target token with padding
    #   input_ids = Nx.tensor([[128_022, 21457, 117, 14906, 8, 37089, 432, 110_309, 10550, 2]])
    #   attention_mask = Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    #   decoder_input_ids = Nx.tensor([[128_028, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    #   input = %{
    #     "input_ids" => input_ids,
    #     attention_mask: attention_mask,
    #     decoder_input_ids: decoder_input_ids
    #   }

    #   output = Axon.predict(model, params, input)

    #   assert Nx.shape(output.logits) == {1, 10, 128_112}

    #   assert_all_close(
    #     output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
    #     Nx.tensor([
    #       [[-2.2073, 11.7937, 1.9133], [-2.1376, 12.3877, 2.5157], [-2.1423, 12.6261, 2.7610]]
    #     ]),
    #     atol: 1.0e-4
    #   )
    # end
  end
end
