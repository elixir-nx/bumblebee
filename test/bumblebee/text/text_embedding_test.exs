defmodule Bumblebee.Text.TextEmbeddingTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "returns E5 embedding for a piece of text" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "intfloat/e5-large"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "intfloat/e5-large"})

      serving = Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer)

      text = "query: Cats are cute."

      assert Nx.shape(Nx.Serving.run(serving, text).embedding) == {1024}
    end
  end
end
