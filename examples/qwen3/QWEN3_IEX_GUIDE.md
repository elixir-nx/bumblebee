# Qwen3 IEx Usage Guide

## Text Generation (Qwen3-4B-Instruct)

```elixir
# Start IEx
iex -S mix

# Set backend
Nx.default_backend(EXLA.Backend)

# Load model components
{:ok, m} = Bumblebee.load_model({:hf, "Qwen/Qwen3-4B-Instruct-2507"})
{:ok, t} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-4B-Instruct-2507"})
{:ok, c} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-4B-Instruct-2507"})

# Create serving
s = Bumblebee.Text.generation(m, t, c)

# Generate text
Nx.Serving.run(s, "The future of AI is")

# With chat format
prompt = "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is Elixir?<|im_end|>
<|im_start|>assistant
"
Nx.Serving.run(s, prompt)
```

## Text Embeddings (Qwen3-Embedding-0.6B)

### Method 1: Using :for_embedding Architecture (Recommended)

```elixir
# Start IEx
iex -S mix

# Set backend
Nx.default_backend(EXLA.Backend)

# Load embedding model with :for_embedding architecture
{:ok, m} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-0.6B"},
  architecture: :for_embedding
)
{:ok, t} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-Embedding-0.6B"})

# Create serving
s = Bumblebee.Text.text_embedding(m, t,
  output_attribute: :embedding,
  embedding_processor: :l2_norm
)

# Generate embeddings
e1 = Nx.Serving.run(s, "The cat sat on the mat")
e2 = Nx.Serving.run(s, "A feline rested on the rug")
e3 = Nx.Serving.run(s, "Python is a programming language")

# Check dimension
Nx.shape(e1.embedding)  # {1024}

# Compute similarity
Nx.dot(e1.embedding, e2.embedding) |> Nx.to_number()  # ~0.73 (similar)
Nx.dot(e1.embedding, e3.embedding) |> Nx.to_number()  # ~0.34 (different)
```

### Method 2: Direct Model Access (Advanced)

```elixir
# For more control over the pipeline
{:ok, m} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-0.6B"},
  architecture: :for_embedding
)
{:ok, t} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-Embedding-0.6B"})

{_init, predict} = Axon.build(m.model)

# Generate embedding
inputs = Bumblebee.apply_tokenizer(t, "test text")
output = predict.(m.params, inputs)
embedding = Bumblebee.Utils.Nx.normalize(output.embedding)
Nx.shape(embedding)  # {1, 1024}
```

## Instruction-Aware Embeddings (Qwen Team Recommendation)

```elixir
# Setup
Nx.default_backend(EXLA.Backend)
{:ok, m} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-0.6B"},
  architecture: :for_embedding
)
{:ok, t} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-Embedding-0.6B"})
s = Bumblebee.Text.text_embedding(m, t,
  output_attribute: :embedding,
  embedding_processor: :l2_norm
)

# Without instruction
query = "What is the capital of France?"
q_plain = Nx.Serving.run(s, query)

# With instruction (recommended by Qwen team)
query_prompted = "Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: What is the capital of France?"
q_with_prompt = Nx.Serving.run(s, query_prompted)

# Documents (no instruction needed)
doc = "Paris is the capital and largest city of France."
d = Nx.Serving.run(s, doc)

# Compare
Nx.dot(q_plain.embedding, d.embedding) |> Nx.to_number()
Nx.dot(q_with_prompt.embedding, d.embedding) |> Nx.to_number()
```

## Custom Task Instructions

```elixir
# Code search
code_query = "Instruct: Given a code search query, find relevant code snippets
Query: function to calculate factorial"

code_doc = "def factorial(n), do: if n <= 1, do: 1, else: n * factorial(n - 1)"

q = Nx.Serving.run(s, code_query)
d = Nx.Serving.run(s, code_doc)

Nx.dot(q.embedding, d.embedding) |> Nx.to_number()  # High similarity
```

## Semantic Search Example

```elixir
# Index documents
documents = [
  "Paris is the capital of France",
  "Berlin is the capital of Germany",
  "Machine learning uses neural networks",
  "The Eiffel Tower is in Paris"
]

doc_embeddings = Enum.map(documents, fn doc ->
  Nx.Serving.run(s, doc).embedding
end)

# Search
query = "Instruct: Given a web search query, retrieve relevant passages
Query: What is the French capital?"
q_emb = Nx.Serving.run(s, query).embedding

# Compute similarities
similarities = Enum.map(doc_embeddings, fn doc_emb ->
  Nx.dot(q_emb, doc_emb) |> Nx.to_number()
end)

# Show results ranked by similarity
Enum.zip(documents, similarities)
|> Enum.sort_by(&elem(&1, 1), :desc)
|> Enum.each(fn {doc, score} ->
  IO.puts("#{Float.round(score, 3)}: #{doc}")
end)
```

## Batch Processing

```elixir
# Process multiple texts at once
texts = [
  "First document",
  "Second document",
  "Third document"
]

results = Nx.Serving.run(s, texts)

embeddings = Enum.map(results, & &1.embedding)
```

## Model Variants

```elixir
# Different sizes available
{:ok, m} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-0.6B"}, architecture: :for_embedding)
{:ok, m} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-4B"}, architecture: :for_embedding)
{:ok, m} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Embedding-8B"}, architecture: :for_embedding)
```

## Common Similarity Metrics

```elixir
# Cosine similarity (recommended for normalized embeddings)
cosine_sim = fn e1, e2 -> Nx.dot(e1, e2) |> Nx.to_number() end

# Euclidean distance
euclidean = fn e1, e2 ->
  Nx.subtract(e1, e2) |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number()
end

# Manhattan distance
manhattan = fn e1, e2 ->
  Nx.subtract(e1, e2) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
end
```
