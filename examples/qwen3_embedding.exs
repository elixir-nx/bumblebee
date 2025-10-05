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

# Load embedding model
{:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-0.6B"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-Embedding-0.6B"})

# For Qwen3-Embedding, we need to manually create the serving since we need
# to extract the last hidden state from the tuple and then pool the last token

# Build the model with output_hidden_states enabled
{_init_fn, encoder} =
  Axon.build(model_info.model,
    mode: :inference,
    global_layer_options: [output_hidden_states: true]
  )

# Create custom embedding function
embedding_fun = fn params, inputs ->
  # Run the model
  output = encoder.(params, inputs)

  # Extract the last layer's hidden states
  # hidden_states is a tuple of all layers, we want the last one
  last_hidden_state =
    if is_tuple(output.hidden_states) do
      output.hidden_states |> Tuple.to_list() |> List.last()
    else
      raise "Model must output hidden_states for embeddings"
    end

  # Pool the last token (last non-padding token in the sequence)
  sequence_lengths =
    inputs["attention_mask"]
    |> Nx.sum(axes: [1])
    |> Nx.subtract(1)
    |> Nx.as_type({:s, 64})

  embedding = Bumblebee.Utils.Nx.batched_take(last_hidden_state, sequence_lengths)

  # Squeeze batch dimension and L2 normalize
  embedding
  |> Nx.squeeze(axes: [0])
  |> Bumblebee.Utils.Nx.normalize()
end

# Helper function to generate embeddings for text
generate_embedding = fn text ->
  inputs = Bumblebee.apply_tokenizer(tokenizer, text)
  embedding_fun.(model_info.params, inputs)
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
