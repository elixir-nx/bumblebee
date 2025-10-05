# Bumblebee Examples

This directory contains example scripts demonstrating how to use Bumblebee models.

## Qwen3 Examples

See the `qwen3/` subdirectory for comprehensive Qwen3 model examples:

### Text Generation
```bash
elixir examples/qwen3/qwen3.exs
```

### Text Embeddings
```bash
elixir examples/qwen3/qwen3_embedding.exs
elixir examples/qwen3/qwen3_embedding_prompts.exs
```

### Document Reranking
```bash
elixir examples/qwen3/qwen3_reranker.exs
```

### Features Demonstrated

**Text Generation** (`qwen3.exs`):
- Text completion
- Question answering
- Chat format
- Code generation

**Embeddings** (`qwen3_embedding.exs`, `qwen3_embedding_prompts.exs`):
- 1024-dimensional text embeddings
- Semantic similarity computation
- Instruction-aware prompts (recommended by Qwen team)
- Multilingual support
- Code search

**Reranking** (`qwen3_reranker.exs`):
- Query-document relevance scoring
- Custom task instructions
- Top-k result selection

### Requirements

- **Text Generation**: ~8GB disk space, ~10GB RAM
- **Embeddings**: ~1.5GB disk space, ~4GB RAM (0.6B model)
- **Reranking**: ~1.5GB disk space, ~4GB RAM (0.6B model)
- **Backend**: EXLA (CPU or GPU)

### Documentation

See `examples/qwen3/QWEN3_IEX_GUIDE.md` for interactive IEx usage examples.

## Phoenix Examples

See the `phoenix/` subdirectory for LiveView-based examples.
