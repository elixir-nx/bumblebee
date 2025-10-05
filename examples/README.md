# Bumblebee Examples

This directory contains example scripts demonstrating how to use Bumblebee models.

## Qwen3 Text Generation

### Basic Usage

```bash
elixir examples/qwen3_text_generation.exs
```

This example demonstrates:
- Loading Qwen3-4B-Instruct model
- Text completion
- Question answering
- Story generation
- Chat format (Instruct model)
- Code generation

### Requirements

- **Disk space**: ~8GB for model weights (downloaded once and cached)
- **Memory**: ~10GB RAM for inference
- **Backend**: EXLA (CPU or GPU)

### Example Output

```
=== Example 1: Text Completion ===
The future of artificial intelligence is being shaped by the development
of more advanced models that can understand and generate human-like language...

=== Example 2: Question Answering ===
What are the benefits of functional programming? The main benefits are
immutability, composability, and easier testing...
```

### Customization

Edit the script to:
- Change `max_new_tokens` for longer/shorter output
- Adjust `temperature` (0.0-1.0) for more deterministic/creative output
- Modify `top_k` and `top_p` for sampling behavior
- Use different prompts

### Other Models

To use different Qwen3 model sizes, change the model name:

```elixir
# Smaller (faster)
{:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})

# Balanced (recommended)
{:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-4B-Instruct-2507"})

# Larger (better quality)
{:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-8B"})
```

## Phoenix Examples

See the `phoenix/` subdirectory for LiveView-based examples.
