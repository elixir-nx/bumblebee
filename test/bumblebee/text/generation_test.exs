defmodule Bumblebee.Text.GenerationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test "decoder model" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    assert %Bumblebee.Text.Gpt2{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[0, 0, 10, 20, 30, 40, 50, 60, 70, 80]]),
      "attention_mask" => Nx.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]),
      "seed" => Nx.tensor([0])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 3)

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[80, 80, 80]]))
  end

  test "encoder-decoder model" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"}
             )

    {:ok, generation_config} =
      Bumblebee.load_generation_config(
        {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"}
      )

    assert %Bumblebee.Text.Bart{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "seed" => Nx.tensor([0])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 3)

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[988, 988, 988]]))
  end

  test "encoder-decoder model and lower precision" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"},
               type: :f16
             )

    {:ok, generation_config} =
      Bumblebee.load_generation_config(
        {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"}
      )

    assert %Bumblebee.Text.Bart{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "seed" => Nx.tensor([0])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 3)

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[988, 988, 988]]))
  end
end
