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

      assert %Nx.Tensor{} = embedding = Nx.Serving.run(serving, text)

      assert Nx.shape(embedding) == {1024}

      assert_all_close(
        embedding[1..3],
        Nx.tensor([-0.9815, -0.5015, 0.9868]),
        atol: 1.0e-4
      )
    end

    test "returns normalized E5 embedding for a piece of text" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "intfloat/e5-large"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "intfloat/e5-large"})

      options = [embedding_processor: :l2_norm]

      serving = Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer, options)

      text = "query: Cats are cute."

      assert %Nx.Tensor{} = embedding = Nx.Serving.run(serving, text)

      assert Nx.shape(embedding) == {1024}

      assert_all_close(
        embedding[1..3],
        Nx.tensor([-0.0459, -0.0234, 0.0461]),
        atol: 1.0e-4
      )

      assert_equal(Nx.sum(Nx.pow(embedding, 2)), Nx.tensor(1.0))
    end
  end
end
