#!/usr/bin/env elixir

# Qwen3-Reranker Example
#
# This example demonstrates using Qwen3-Reranker-0.6B for reranking
# documents based on relevance to a query. Rerankers score query-document
# pairs to improve retrieval quality in RAG and search applications.
#
# Usage: elixir examples/qwen3_reranker.exs

Mix.install([
  {:bumblebee, path: Path.expand("..", __DIR__)},
  {:exla, ">= 0.0.0"}
])

Application.put_env(:nx, :default_backend, EXLA.Backend)

# Load reranker model (uses same Qwen3ForCausalLM architecture)
{:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Reranker-0.6B"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-Reranker-0.6B"})

# Build model
{_init_fn, predict_fn} = Axon.build(model_info.model)

# Get yes/no token IDs by tokenizing the words
tokenizer_no_special = Bumblebee.configure(tokenizer, add_special_tokens: false)
yes_token_result = Bumblebee.apply_tokenizer(tokenizer_no_special, "yes")
no_token_result = Bumblebee.apply_tokenizer(tokenizer_no_special, "no")
yes_token_id = Nx.to_flat_list(yes_token_result["input_ids"]) |> hd()
no_token_id = Nx.to_flat_list(no_token_result["input_ids"]) |> hd()

# Format query-document pair as recommended by Qwen team
format_pair = fn instruction, query, document ->
  instruction =
    instruction || "Given a web search query, retrieve relevant passages that answer the query"

  # Format with suffix as per vLLM example
  suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  "<Instruct>: #{instruction}\n<Query>: #{query}\n<Document>: #{document}#{suffix}"
end

# Reranking function
# Returns a relevance score between 0 and 1
rerank_score = fn query, document, instruction ->
  # Format the input
  text = format_pair.(instruction, query, document)

  # Tokenize
  inputs = Bumblebee.apply_tokenizer(tokenizer, text)

  # Get model output
  output = predict_fn.(model_info.params, inputs)

  # Extract logits for the last token
  # Shape: {batch, seq, vocab}
  {_batch, seq_len, _vocab} = Nx.shape(output.logits)
  last_logits = output.logits[[0, seq_len - 1, ..]]

  # Get logits for "yes" and "no" tokens
  yes_logit = Nx.to_number(last_logits[yes_token_id])
  no_logit = Nx.to_number(last_logits[no_token_id])

  # Compute softmax probability for "yes"
  # P(yes) = exp(yes) / (exp(yes) + exp(no))
  exp_yes = :math.exp(yes_logit)
  exp_no = :math.exp(no_logit)

  relevance_score = exp_yes / (exp_yes + exp_no)
  relevance_score
end

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("Qwen3-Reranker Example")
IO.puts(String.duplicate("=", 70))

# Example 1: Basic reranking
IO.puts("\n=== Example 1: Basic Query-Document Scoring ===")

query = "What is the capital of France?"

documents = [
  "Paris is the capital and largest city of France.",
  "London is the capital of the United Kingdom.",
  "Machine learning is a subset of artificial intelligence.",
  "The Eiffel Tower is located in Paris, France.",
  "Berlin is the capital of Germany."
]

IO.puts("Query: #{query}")
IO.puts("\nDocument relevance scores:")

scores =
  Enum.map(documents, fn doc ->
    score = rerank_score.(query, doc, nil)
    {doc, score}
  end)
  |> Enum.sort_by(&elem(&1, 1), :desc)

Enum.with_index(scores, 1)
|> Enum.each(fn {{doc, score}, rank} ->
  IO.puts("  #{rank}. [#{Float.round(score, 4)}] #{String.slice(doc, 0..60)}...")
end)

# Example 2: Custom instruction
IO.puts("\n=== Example 2: Custom Task Instruction ===")

instruction = "Given a coding question, find relevant code examples"
query = "How to calculate factorial in Elixir?"

code_docs = [
  "def factorial(n), do: if n <= 1, do: 1, else: n * factorial(n - 1)",
  "def fibonacci(n), do: if n <= 1, do: n, else: fibonacci(n - 1) + fibonacci(n - 2)",
  "def sum_list(list), do: Enum.reduce(list, 0, &+/2)",
  "Factorial is a mathematical function that multiplies a number by all positive integers less than it."
]

IO.puts("Query: #{query}")
IO.puts("Instruction: #{instruction}")
IO.puts("\nCode snippet relevance:")

code_docs
|> Enum.map(fn doc ->
  score = rerank_score.(query, doc, instruction)
  {doc, score}
end)
|> Enum.sort_by(&elem(&1, 1), :desc)
|> Enum.with_index(1)
|> Enum.each(fn {{doc, score}, rank} ->
  IO.puts("  #{rank}. [#{Float.round(score, 4)}] #{String.slice(doc, 0..60)}...")
end)

# Example 3: Reranking search results
IO.puts("\n=== Example 3: Reranking Initial Search Results ===")

query = "best practices for concurrent programming"

# Simulated initial retrieval results (could be from vector search)
search_results = [
  "Concurrent programming involves multiple computations executing simultaneously.",
  "Elixir uses the Actor model for concurrency with lightweight processes.",
  "Python has threading and multiprocessing modules for parallel execution.",
  "The weather is nice today and perfect for a walk.",
  "OTP behaviors like GenServer provide patterns for concurrent systems."
]

IO.puts("Query: #{query}")
IO.puts("\nInitial results (unranked):")

Enum.with_index(search_results, 1)
|> Enum.each(fn {doc, i} ->
  IO.puts("  #{i}. #{String.slice(doc, 0..60)}...")
end)

IO.puts("\nAfter reranking:")

search_results
|> Enum.map(fn doc ->
  score = rerank_score.(query, doc, nil)
  {doc, score}
end)
|> Enum.sort_by(&elem(&1, 1), :desc)
# Top 3 results
|> Enum.take(3)
|> Enum.with_index(1)
|> Enum.each(fn {{doc, score}, rank} ->
  IO.puts("  #{rank}. [#{Float.round(score, 4)}] #{String.slice(doc, 0..60)}...")
end)

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("✓ Qwen3-Reranker successfully reranked documents by relevance")
IO.puts(String.duplicate("=", 70) <> "\n")
