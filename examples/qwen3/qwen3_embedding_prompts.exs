#!/usr/bin/env elixir

# Qwen3-Embedding with Instruction Prompts
#
# This example demonstrates the Qwen team's recommended approach for using
# instruction-aware prompts to improve retrieval performance by 1-5%.
#
# Usage: elixir examples/qwen3_embedding_prompts.exs

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

# Create serving with L2 normalization
serving =
  Bumblebee.Text.text_embedding(model_info, tokenizer,
    output_attribute: :embedding,
    embedding_processor: :l2_norm
  )

# Embedding function
generate_embedding = fn text ->
  result = Nx.Serving.run(serving, text)
  result.embedding
end

# Helper to compute similarity
similarity = fn e1, e2 ->
  Nx.dot(e1, e2) |> Nx.to_number() |> Float.round(4)
end

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("Qwen3-Embedding: With vs Without Instruction Prompts")
IO.puts(String.duplicate("=", 70))

# Test data
query_text = "What is the capital of France?"
document1 = "Paris is the capital and largest city of France."
document2 = "London is the capital of the United Kingdom."
document3 = "Machine learning is a branch of artificial intelligence."

IO.puts("\nQuery: #{query_text}")
IO.puts("Doc 1: #{document1}")
IO.puts("Doc 2: #{document2}")
IO.puts("Doc 3: #{document3}")

# ==============================================================================
# Test 1: WITHOUT instruction prompts (baseline)
# ==============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("TEST 1: Without Instruction Prompts (Baseline)")
IO.puts(String.duplicate("-", 70))

q_plain = generate_embedding.(query_text)
d1_plain = generate_embedding.(document1)
d2_plain = generate_embedding.(document2)
d3_plain = generate_embedding.(document3)

IO.puts("\nSimilarity scores:")
IO.puts("  Query vs Doc1 (relevant):     #{similarity.(q_plain, d1_plain)}")
IO.puts("  Query vs Doc2 (semi-relevant): #{similarity.(q_plain, d2_plain)}")
IO.puts("  Query vs Doc3 (irrelevant):    #{similarity.(q_plain, d3_plain)}")

# ==============================================================================
# Test 2: WITH instruction prompts (Qwen team's recommendation)
# ==============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("TEST 2: With Instruction Prompts (Qwen Team Recommendation)")
IO.puts(String.duplicate("-", 70))

# Qwen team's recommended format for web search
query_with_prompt =
  "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: #{query_text}"

# Documents don't need prompts (as per config)
q_prompted = generate_embedding.(query_with_prompt)

IO.puts("\nQuery prompt format:")
IO.puts("  Instruct: Given a web search query, retrieve relevant passages...")
IO.puts("  Query: #{query_text}")

IO.puts("\nSimilarity scores:")
IO.puts("  Query vs Doc1 (relevant):     #{similarity.(q_prompted, d1_plain)}")
IO.puts("  Query vs Doc2 (semi-relevant): #{similarity.(q_prompted, d2_plain)}")
IO.puts("  Query vs Doc3 (irrelevant):    #{similarity.(q_prompted, d3_plain)}")

# ==============================================================================
# Test 3: Custom task instructions
# ==============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("TEST 3: Custom Task Instructions")
IO.puts(String.duplicate("-", 70))

# Code search example
code_query = "function to calculate factorial"

code_docs = [
  "def factorial(n), do: if n <= 1, do: 1, else: n * factorial(n - 1)",
  "def fibonacci(n), do: if n <= 1, do: n, else: fibonacci(n - 1) + fibonacci(n - 2)",
  "defmodule Calculator do; def add(a, b), do: a + b; end"
]

code_query_prompt =
  "Instruct: Given a code search query, find relevant code snippets\nQuery: #{code_query}"

IO.puts("\nCode Search Task:")
IO.puts("Query: #{code_query}")

q_code = generate_embedding.(code_query_prompt)
code_embeddings = Enum.map(code_docs, &generate_embedding.(&1))

Enum.zip(code_docs, code_embeddings)
|> Enum.with_index(1)
|> Enum.each(fn {{doc, emb}, idx} ->
  sim = similarity.(q_code, emb)
  IO.puts("  Code #{idx} similarity: #{sim}")
  IO.puts("    #{String.slice(doc, 0..60)}...")
end)

# ==============================================================================
# Test 4: Multilingual example
# ==============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("TEST 4: Multilingual Embeddings")
IO.puts(String.duplicate("-", 70))

multilingual_texts = [
  "The cat is sleeping",
  "El gato está durmiendo",
  "Le chat dort",
  "猫在睡觉",
  "The dog is running"
]

IO.puts("\nGenerating embeddings for 5 texts in different languages...")
multi_embeddings = Enum.map(multilingual_texts, &generate_embedding.(&1))

IO.puts("\nSemantic similarity (all about cat sleeping vs dog running):")

Enum.take(multi_embeddings, 4)
|> Enum.with_index(1)
|> Enum.each(fn {emb, idx} ->
  sim_to_english = similarity.(hd(multi_embeddings), emb)
  sim_to_dog = similarity.(List.last(multi_embeddings), emb)
  IO.puts("  Text #{idx}: same_meaning=#{sim_to_english}, different=#{sim_to_dog}")
end)

# ==============================================================================
# Summary
# ==============================================================================

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("SUMMARY")
IO.puts(String.duplicate("=", 70))
IO.puts("✓ Qwen3-Embedding supports instruction-aware prompts")
IO.puts("✓ Recommended format: 'Instruct: [task]\\nQuery: [query]'")
IO.puts("✓ Improves retrieval performance by 1-5%")
IO.puts("✓ Works for multilingual and code search tasks")
IO.puts("✓ Generates 1024-dimensional normalized vectors")
IO.puts(String.duplicate("=", 70) <> "\n")
