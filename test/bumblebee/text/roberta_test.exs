defmodule Bumblebee.Text.RoBERTaTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :slow
    @tag :capture_log
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "roberta-base"}, architecture: :base)

      assert %Bumblebee.Text.Roberta{architecture: :base} = config

      input = %{
        "input_ids" => Nx.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.last_hidden_state) == {1, 11, 768}

      assert_all_close(
        output.last_hidden_state[[0..-1//1, 0..2, 0..2]],
        Nx.tensor([
          [[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]
        ]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "masked language modeling model ids" do
      assert {:ok, model, params, config} = Bumblebee.load_model({:hf, "roberta-base"})

      assert %Bumblebee.Text.Roberta{architecture: :for_masked_language_modeling} = config

      input = %{
        "input_ids" => Nx.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 11, 50265}

      assert_all_close(
        output.logits[[0..-1//1, 0..2, 0..2]],
        Nx.tensor([
          [[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]
        ]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "casual language modeling model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "roberta-base"},
                 architecture: :for_causal_language_modeling
               )

      assert %Bumblebee.Text.Roberta{architecture: :for_causal_language_modeling} = config

      input = %{
        "input_ids" => Nx.tensor([[0, 234, 546, 3218, 54, 1544, 15856, 2]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 8, 50265}

      assert_all_close(
        output.logits[[0..-1//1, 0..2, 0..2]],
        Nx.tensor([
          [[32.8680, -4.4621, 20.4998], [2.8034, -4.3022, 10.9247], [-1.3060, -4.5799, 6.5772]]
        ]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag timeout: 200_000
    @tag :capture_log
    test "sequence classification" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "cardiffnlp/twitter-roberta-base-emotion"})

      assert %Bumblebee.Text.Roberta{architecture: :for_sequence_classification} = config

      input = %{
        "input_ids" => Nx.tensor([[0, 31414, 6, 127, 2335, 16, 11962, 37, 11639, 1168, 2]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 4}

      assert_all_close(
        output.logits,
        Nx.tensor([[-1.3661, 3.0174, -0.9609, -0.4145]]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag timeout: 400_000
    @tag :capture_log
    test "multiple choice model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "LIAMF-USP/aristo-roberta"})

      assert %Bumblebee.Text.Roberta{architecture: :for_multiple_choice} = config

      input = %{
        "input_ids" =>
          Nx.tensor([
            [[0, 38576, 103, 4437, 2, 2, 725, 895, 2], [0, 38576, 103, 4437, 2, 2, 487, 895, 2]]
          ]),
        "attention_mask" =>
          Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 2}

      assert_all_close(
        output.logits,
        Nx.tensor([[-13.9123, -13.4582]]),
        atol: 1.0e-3
      )
    end

    @tag :slow
    @tag timeout: 400_000
    @tag :capture_log
    test "token classification model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "Jean-Baptiste/roberta-large-ner-english"})

      assert %Bumblebee.Text.Roberta{architecture: :for_token_classification} = config

      input = %{
        "input_ids" => Nx.tensor([[30581, 3923, 34892, 16, 10, 138, 716, 11, 2201, 8, 188, 469]])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.logits) == {1, 12, 5}

      assert_all_close(
        output.logits[[0..-1//1, 0..2, 0..1]],
        Nx.tensor([[[4.1969, -2.5614], [-1.4174, -0.6959], [-1.3807, 0.1313]]]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag timeout: 200_000
    @tag :capture_log
    test "question answering model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"})

      assert %Bumblebee.Text.Roberta{architecture: :for_question_answering} = config

      input = %{
        "input_ids" =>
          Nx.tensor([
            [0, 12375, 21, 2488, 289, 13919, 116, 2, 2, 24021, 289, 13919, 21, 10, 2579, 29771, 2]
          ])
      }

      output = Axon.predict(model, params, input)

      assert Nx.shape(output.start_logits) == {1, 17}
      assert Nx.shape(output.end_logits) == {1, 17}

      assert_all_close(
        output.start_logits[[0..-1//1, 0..2]],
        Nx.tensor([[0.5901, -8.3490, -8.8031]]),
        atol: 1.0e-4
      )

      assert_all_close(
        output.end_logits[[0..-1//1, 0..2]],
        Nx.tensor([[1.1207, -7.5968, -7.6151]]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    @tag :capture_log
    test "masked language modeling model" do
      assert {:ok, model, params, config} = Bumblebee.load_model({:hf, "roberta-base"})
      assert %Bumblebee.Text.Roberta{architecture: :for_masked_language_modeling} = config

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "roberta-base"})

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, "The current capital city of France is <mask>.")

      output = Axon.predict(model, params, inputs)

      assert Nx.shape(output.logits) == {1, 11, 50265}
      mask_token_id = Bumblebee.Tokenizer.special_token_id(tokenizer, :mask)

      mask_idx = inputs["input_ids"] |> Nx.equal(mask_token_id) |> Nx.argmax()
      id = output.logits[[0, mask_idx]] |> Nx.argmax() |> Nx.to_number()

      assert Bumblebee.Tokenizer.decode(tokenizer, id) == {:ok, " Paris"}
    end
  end
end
