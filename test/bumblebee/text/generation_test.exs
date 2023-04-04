defmodule Bumblebee.Text.GenerationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "generates text" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/bart-large-cnn"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-cnn"})

      article = """
      PG&E stated it scheduled the blackouts in response to forecasts for high \
      winds amid dry conditions. The aim is to reduce the risk of wildfires. \
      Nearly 800 thousand customers were scheduled to be affected by the shutoffs \
      which were expected to last through at least midday tomorrow.
      """

      serving =
        Bumblebee.Text.generation(model_info, tokenizer,
          max_new_tokens: 8,
          defn_options: [compiler: EXLA]
        )

      assert %{results: [%{text: "PG&E scheduled the black"}]} = Nx.Serving.run(serving, article)
    end

    test "with :no_repeat_ngram_length" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})

      serving =
        Bumblebee.Text.generation(model_info, tokenizer,
          max_new_tokens: 12,
          no_repeat_ngram_length: 2,
          defn_options: [compiler: EXLA]
        )

      # Without :no_repeat_ngram_length we get
      # %{results: [%{text: "I was going to say, 'Well, I'm going to say,"}]}

      assert %{results: [%{text: "I was going to say, 'Well, I'm going back to the"}]} =
               Nx.Serving.run(serving, "I was going")
    end

    test "contrastive search" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})

      serving =
        Bumblebee.Text.generation(model_info, tokenizer,
          max_new_tokens: 12,
          top_k: 4,
          penalty_alpha: 0.6,
          defn_options: [compiler: EXLA]
        )

      assert %{results: [%{text: "I was going to say, 'Well, I don't know what you"}]} =
               Nx.Serving.run(serving, "I was going")
    end
  end
end
