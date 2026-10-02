defmodule Bumblebee.Text.EmbeddingGemmaIntegrationTest do
  use ExUnit.Case, async: false

  @moduletag :slow
  @moduletag timeout: 600_000

  # Opt in with BUMBLEBEE_EMBEDDING_GEMMA=/path/to/checkpoint and --include slow.
  # The gated Google checkpoint must already be available locally.
  if path = System.get_env("BUMBLEBEE_EMBEDDING_GEMMA") do
    import Bumblebee.TestHelpers

    @path path

    test "EmbeddingGemma transformer, tokenizer and embedding serving" do
      repository = {:local, @path}
      assert {:ok, model_info} = Bumblebee.load_model(repository, architecture: :base)
      assert model_info.spec.use_bidirectional_attention
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer(repository)

      serving =
        Bumblebee.Text.text_embedding(model_info, tokenizer,
          output_attribute: :hidden_state,
          output_pool: :mean_pooling,
          embedding_processor: :l2_norm
        )

      short = "task: search result | title: Example | text: A bee visits a flower."
      long = "task: search result | title: Example | text: Bees collect pollen and make honey."
      %{embedding: single} = Nx.Serving.run(serving, short)
      [first, second] = Nx.Serving.run(serving, [short, long])

      assert Nx.shape(single) == {model_info.spec.hidden_size}
      assert_all_close(first.embedding, single, atol: 1.0e-4, rtol: 1.0e-4)

      for embedding <- [first.embedding, second.embedding] do
        assert_all_close(Nx.sum(Nx.pow(embedding, 2)), Nx.tensor(1.0))
      end
    end
  end
end
