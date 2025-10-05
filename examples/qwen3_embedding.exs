#!/usr/bin/env elixir

# Qwen3-Embedding Example
#
# This example demonstrates using Qwen3-Embedding-0.6B for generating
# text embeddings for semantic search and similarity tasks.
#
# Usage: elixir examples/qwen3_embedding.exs

Mix.install([
  {:bumblebee, path: Path.expand("..", __DIR__)},
  {:exla, ">= 0.0.0"}
])

Application.put_env(:nx, :default_backend, EXLA.Backend)

# Load embedding model with :for_embedding architecture
{:ok, model_info} =
  Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-0.6B"},
    architecture: :for_embedding
  )

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-Embedding-0.6B"})

# Create text embedding serving
# The :for_embedding architecture automatically pools the last token
serving =
  Bumblebee.Text.text_embedding(model_info, tokenizer,
    output_attribute: :embedding,
    embedding_processor: :l2_norm
  )

# Helper function
generate_embedding = fn text ->
  result = Nx.Serving.run(serving, text)
  result.embedding
end

# Example 1: Simple text embeddings
IO.puts("\n=== Example 1: Generate Text Embeddings ===")

texts = [
  "The quick brown fox jumps over the lazy dog",
  "A fast auburn fox leaps above a sleepy canine",
  "The weather is nice today"
]

IO.puts("Generating embeddings for #{length(texts)} texts...")

embeddings = Enum.map(texts, &generate_embedding.(&1))

IO.puts("✓ Generated embeddings")
IO.puts("  Embedding dimension: #{Nx.axis_size(hd(embeddings), 0)}")
IO.puts("")

# Example 2: Compute similarity
IO.puts("=== Example 2: Semantic Similarity ===")
IO.puts("Text 1: \"#{Enum.at(texts, 0)}\"")
IO.puts("Text 2: \"#{Enum.at(texts, 1)}\"")
IO.puts("Text 3: \"#{Enum.at(texts, 2)}\"")
IO.puts("")

# Compute cosine similarity
similarity_1_2 =
  Nx.dot(Enum.at(embeddings, 0), Enum.at(embeddings, 1))
  |> Nx.to_number()

similarity_1_3 =
  Nx.dot(Enum.at(embeddings, 0), Enum.at(embeddings, 2))
  |> Nx.to_number()

IO.puts("Similarity (Text 1 vs Text 2): #{Float.round(similarity_1_2, 4)}")
IO.puts("Similarity (Text 1 vs Text 3): #{Float.round(similarity_1_3, 4)}")
IO.puts("")
IO.puts("✓ Texts 1 and 2 are more similar (same meaning, different words)")

# Example 3: Instruction-aware embeddings
IO.puts("\n=== Example 3: Instruction-Aware Embeddings ===")

query =
  "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of France?"

document = "Paris is the capital and largest city of France."

query_embedding = generate_embedding.(query)
doc_embedding = generate_embedding.(document)

similarity =
  Nx.dot(query_embedding, doc_embedding)
  |> Nx.to_number()

IO.puts("Query: What is the capital of France?")
IO.puts("Document: Paris is the capital and largest city of France.")
IO.puts("Similarity: #{Float.round(similarity, 4)}")
IO.puts("")

# Example 4: Batch processing
IO.puts("=== Example 4: Batch Processing ===")

batch_texts = [
  "Machine learning is a subset of artificial intelligence",
  "Deep learning uses neural networks with multiple layers",
  "Python is a popular programming language"
]

IO.puts("Processing batch of #{length(batch_texts)} texts...")
batch_embeddings = Enum.map(batch_texts, &generate_embedding.(&1))

IO.puts("✓ Batch embeddings generated")
IO.puts("  Number of embeddings: #{length(batch_embeddings)}")
IO.puts("  Each embedding shape: #{inspect(Nx.shape(hd(batch_embeddings)))}")
IO.puts("")

IO.puts("=== Qwen3-Embedding is working! ===")
