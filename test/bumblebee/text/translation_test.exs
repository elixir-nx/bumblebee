defmodule Bumblebee.Text.TranslationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  test "generates text with greedy generation" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/nllb-200-distilled-600M"})

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/nllb-200-distilled-600M"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "facebook/nllb-200-distilled-600M"})

    serving = Bumblebee.Text.translation(model_info, tokenizer, generation_config)

    text = "The bank of the river is beautiful in spring"

    assert %{
             results: [
               %{
                 text: "W wiosnę brzeg rzeki jest piękny",
                 token_summary: %{input: 11, output: 13, padding: 0}
               }
             ]
           } =
             Nx.Serving.run(serving, %{
               text: text,
               source_language_token: "eng_Latn",
               target_language_token: "pol_Latn"
             })
  end
end
