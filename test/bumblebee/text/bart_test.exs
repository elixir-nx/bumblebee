defmodule Bumblebee.Text.BertTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :slow
    @tag :capture_log
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "facebook/bart-base"}, architecture: :base)

      assert %Bumblebee.Text.Bart{architecture: :base} = config

      input_ids = Nx.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
      attention_mask = Nx.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

      input = %{
        "input_ids" => input_ids,
        "decoder_input_ids" => input_ids,
        "attention_mask" => attention_mask,
        "decoder_attention_mask" => attention_mask,
        "position_ids" => Nx.iota(input_ids, axis: -1),
        "decoder_position_ids" => Nx.iota(input_ids, axis: -1)
      }

      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.last_hidden_state) == {1, 11, 768}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[-0.9478, 2.2893, 0.3704], [-2.0416, 0.8665, 0.4935], [-1.7090, 1.3902, -0.8445]]
        ]),
        atol: 1.0e-4
      )
    end
  end
end
