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

      serving = Bumblebee.Text.generation(model_info, tokenizer, max_new_tokens: 8)

      assert %{results: [%{text: "PG&E scheduled the black"}]} = Nx.Serving.run(serving, article)
    end
  end
end
