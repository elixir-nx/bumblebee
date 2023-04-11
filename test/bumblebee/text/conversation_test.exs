defmodule Bumblebee.Text.ConversationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "generates text" do
      {:ok, model} = Bumblebee.load_model({:hf, "microsoft/DialoGPT-medium"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "gpt2"})

      serving = Bumblebee.Text.conversation(model, tokenizer, generation_config)

      history = nil

      message = "Hey!"

      assert %{text: "Hey !", history: history} =
               Nx.Serving.run(serving, %{text: message, history: history})

      message = "What's up?"

      assert %{text: "Not much .", history: _history} =
               Nx.Serving.run(serving, %{text: message, history: history})
    end
  end
end
