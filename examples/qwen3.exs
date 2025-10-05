#!/usr/bin/env elixir

# Qwen3-4B-Instruct Text Generation
#
# This example demonstrates using the Qwen3-4B-Instruct model for various
# text generation tasks including completion, chat, and code generation.
#
# Usage:
#   elixir examples/qwen3.exs

Mix.install([
  {:bumblebee, "~> 0.6.0"},
  {:exla, ">= 0.0.0"}
])

Application.put_env(:nx, :default_backend, EXLA.Backend)

# Load model, tokenizer, and generation configuration
{:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-4B-Instruct-2507"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-4B-Instruct-2507"})
{:ok, generation_config} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-4B-Instruct-2507"})

# Configure generation parameters
generation_config =
  Bumblebee.configure(generation_config,
    max_new_tokens: 100,
    strategy: %{type: :multinomial_sampling, top_k: 20, top_p: 0.8},
    temperature: 0.7
  )

# Create text generation serving
serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)

# Example 1: Text Completion
IO.puts("\n=== Text Completion ===")
result = Nx.Serving.run(serving, "The future of artificial intelligence")
IO.puts(result.results |> hd() |> Map.get(:text))

# Example 2: Question Answering with Chat Format
IO.puts("\n=== Question Answering ===")

prompt = """
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What are the key features of the Elixir programming language?<|im_end|>
<|im_start|>assistant
"""

result = Nx.Serving.run(serving, prompt)
IO.puts(result.results |> hd() |> Map.get(:text))

# Example 3: Code Generation
IO.puts("\n=== Code Generation ===")

prompt = """
<|im_start|>system
You are an expert Elixir programmer.<|im_end|>
<|im_start|>user
Write a function to calculate the nth Fibonacci number using recursion.<|im_end|>
<|im_start|>assistant
"""

result = Nx.Serving.run(serving, prompt)
IO.puts(result.results |> hd() |> Map.get(:text))

# Example 4: Creative Writing
IO.puts("\n=== Creative Writing ===")

prompt = """
<|im_start|>system
You are a creative storyteller.<|im_end|>
<|im_start|>user
Write the opening paragraph of a science fiction story.<|im_end|>
<|im_start|>assistant
"""

result = Nx.Serving.run(serving, prompt)
IO.puts(result.results |> hd() |> Map.get(:text))
