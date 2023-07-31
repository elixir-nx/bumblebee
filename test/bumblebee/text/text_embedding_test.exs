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

      assert %{embedding: %Nx.Tensor{} = embedding} = Nx.Serving.run(serving, text)

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

      assert %{embedding: %Nx.Tensor{} = embedding} = Nx.Serving.run(serving, text)

      assert Nx.shape(embedding) == {1024}

      assert_all_close(
        embedding[1..3],
        Nx.tensor([-0.0459, -0.0234, 0.0461]),
        atol: 1.0e-4
      )

      assert_equal(Nx.sum(Nx.pow(embedding, 2)), Nx.tensor(1.0))
    end

    test "supports compilation for single or multiple sequence lengths" do
      {:ok, model_info} = Bumblebee.load_model({:hf, "intfloat/e5-large"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "intfloat/e5-large"})

      serving_short =
        Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer,
          compile: [batch_size: 1, sequence_length: 8]
        )

      serving_long =
        Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer,
          compile: [batch_size: 1, sequence_length: 16]
        )

      serving_both =
        Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer,
          compile: [batch_size: 1, sequence_length: [8, 16]]
        )

      short_text = "short text"
      long_text = "definitely much longer text that should exceed 16 tokens"

      assert %{embedding: embedding_short} = Nx.Serving.run(serving_short, short_text)
      assert %{embedding: embedding_long} = Nx.Serving.run(serving_long, long_text)

      assert %{embedding: embedding_short2} = Nx.Serving.run(serving_both, short_text)
      assert %{embedding: embedding_long2} = Nx.Serving.run(serving_both, long_text)

      assert_equal(embedding_short, embedding_short2)
      assert_equal(embedding_long, embedding_long2)
    end
  end
end
