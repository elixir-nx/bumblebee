defmodule Bumblebee.Text.MarianTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "base model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Helsinki-NLP/opus-mt-en-de"}, architecture: :base)

      assert %Bumblebee.Text.Marian{architecture: :base} = spec

      input_ids = Nx.tensor([[10778, 72, 152, 2701, 35, 508, 79, 14, 7296, 19, 402, 23, 41, 0]])

      decoder_input_ids =
        Nx.tensor([
          [
            58100,
            3679,
            34,
            3860,
            1487,
            372,
            1065,
            660,
            46,
            44,
            6,
            65,
            16478,
            28061,
            5084,
            19,
            46,
            50,
            34,
            36838,
            82,
            24,
            43,
            6,
            4935,
            15
          ]
        ])

      inputs = %{
        "input_ids" => input_ids,
        "decoder_input_ids" => decoder_input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 26, 512}

      assert_all_close(
        outputs.hidden_state[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([
          [[1.1446, -1.1168, -0.4515], [1.1183, 0.4136, -0.5005], [-0.1860, 3.5362, 0.4024]]
        ]),
        atol: 1.0e-4
      )
    end

    test "for causal language modeling model" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "Helsinki-NLP/opus-mt-fr-en"})

      assert %Bumblebee.Text.Marian{architecture: :for_causal_language_modeling} = spec

      input_ids = Nx.tensor([[452, 9985, 2, 240, 20, 5253, 32, 137, 5860, 0]])

      inputs = %{
        "input_ids" => input_ids
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 10, 59514}

      assert_all_close(
        outputs.logits[[0..-1//1, 1..3, 1..3]],
        Nx.tensor([[[4.5549, 4.3799, 4.9160], [8.6180, 7.4317, 7.5286], [7.6905, 5.9830, 6.4381]]]),
        atol: 1.0e-4
      )
    end

    # TODO: Marian fast tokenizer upstream
    # test "conditional generation" do
    #   {:ok, model_info} = Bumblebee.load_model({:hf, "Helsinki-NLP/opus-mt-fr-en"})
    #   {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Helsinki-NLP/opus-mt-fr-en"})

    #   assert %Bumblebee.Text.Marian{architecture: :for_causal_language_modeling} = model_info.spec

    #   french = "où est l'arrêt de bus ?"

    #   inputs = Bumblebee.apply_tokenizer(tokenizer, french)

    #   generate =
    #     Bumblebee.Text.Generation.build_generate(model_info.model, model_info.spec,
    #       min_length: 0,
    #       max_length: 8
    #     )

    #   token_ids = EXLA.jit(generate).(model_info.params, inputs)

    #   assert Bumblebee.Tokenizer.decode(tokenizer, token_ids) == ["Where's the bus stop?"]
    # end
  end
end
