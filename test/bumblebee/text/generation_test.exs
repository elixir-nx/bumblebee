defmodule Bumblebee.Text.GenerationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "generates text with greedy generation" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/bart-large-cnn"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-cnn"})

      {:ok, generation_config} =
        Bumblebee.load_generation_config({:hf, "facebook/bart-large-cnn"})

      article = """
      PG&E stated it scheduled the blackouts in response to forecasts for high \
      winds amid dry conditions. The aim is to reduce the risk of wildfires. \
      Nearly 800 thousand customers were scheduled to be affected by the shutoffs \
      which were expected to last through at least midday tomorrow.
      """

      generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

      serving =
        Bumblebee.Text.generation(model_info, tokenizer, generation_config)

      assert %{results: [%{text: "PG&E scheduled the black"}]} = Nx.Serving.run(serving, article)
    end

    test "with :no_repeat_ngram_length" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "gpt2"})

      generation_config =
        Bumblebee.configure(generation_config, max_new_tokens: 12, no_repeat_ngram_length: 2)

      serving =
        Bumblebee.Text.generation(model_info, tokenizer, generation_config)

      # Without :no_repeat_ngram_length we get
      # %{results: [%{text: "I was going to say, 'Well, I'm going to say,"}]}

      assert %{results: [%{text: "I was going to say, 'Well, I'm going back to the"}]} =
               Nx.Serving.run(serving, "I was going")
    end

    test "sampling" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "gpt2"})

      generation_config =
        Bumblebee.configure(generation_config,
          max_new_tokens: 12,
          strategy: %{type: :multinomial_sampling}
        )

      serving =
        Bumblebee.Text.generation(model_info, tokenizer, generation_config, seed: 0)

      # Note that this is just a snapshot test, we do not use any
      # reference value, because of PRNG difference

      assert %{
               results: [
                 %{text: "I was going to fall asleep.\"\n\nThis is not Wallace's fifth"}
               ]
             } = Nx.Serving.run(serving, "I was going")
    end

    test "contrastive search" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "gpt2"})

      generation_config =
        Bumblebee.configure(generation_config,
          max_new_tokens: 12,
          strategy: %{type: :contrastive_search, top_k: 4, alpha: 0.6}
        )

      serving =
        Bumblebee.Text.generation(model_info, tokenizer, generation_config)

      assert %{results: [%{text: "I was going to say, 'Well, I don't know what you"}]} =
               Nx.Serving.run(serving, "I was going")
    end

    test "streaming text chunks" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/bart-large-cnn"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-cnn"})

      {:ok, generation_config} =
        Bumblebee.load_generation_config({:hf, "facebook/bart-large-cnn"})

      article = """
      PG&E stated it scheduled the blackouts in response to forecasts for high \
      winds amid dry conditions. The aim is to reduce the risk of wildfires. \
      Nearly 800 thousand customers were scheduled to be affected by the shutoffs \
      which were expected to last through at least midday tomorrow.
      """

      generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

      serving =
        Bumblebee.Text.generation(model_info, tokenizer, generation_config, stream: true)

      stream = Nx.Serving.run(serving, article)
      assert Enum.to_list(stream) == ["PG&E", " scheduled", " the", " black"]

      # Raises when a batch is given
      assert_raise ArgumentError,
                   "serving only accepts singular input when stream is enabled, call the serving with each input in the batch separately",
                   fn ->
                     Nx.Serving.run(serving, [article])
                   end
    end
  end
end
