# Bumblebee

[![Docs](https://img.shields.io/badge/hex.pm-docs-8e7ce6.svg)](https://hexdocs.pm/bumblebee)
[![Actions Status](https://github.com/livebook-dev/kino_bumblebee/workflows/Test/badge.svg)](https://github.com/elixir-nx/bumblebee/actions)

Bumblebee provides pre-trained Neural Network models on top of [Axon](https://github.com/elixir-nx/axon). It includes integration with [🤗 Models](https://huggingface.co/models), allowing anyone to download and perform Machine Learning tasks with few lines of code.

![Numbat and Bumblebees](.github/images/background.jpg)

## Getting started

The best way to get started with Bumblebee is with [Livebook](https://livebook.dev). [Our announcement video](https://news.livebook.dev/announcing-bumblebee-gpt2-stable-diffusion-and-more-in-elixir-3Op73O) shows how to use Livebook's Smart Cells to perform different Neural Network tasks with few clicks. You can then tweak the code and deploy it.

![](.github/images/kino_bumblebee_token_classification.png)

We also provide single-file examples of running Neural Networks inside your Phoenix (+ LiveView) apps inside the [examples/phoenix](examples/phoenix) folder.

![](.github/images/phx_image_classification.png)

You may also check [our official docs](https://hexdocs.pm/bumblebee), which includes notebooks and our API reference. The "Tasks" section in the sidebar covers high-level APIs for using Bumblebee. The remaining modules in the sidebar lists all supported architectures.

## Installation

First add Bumblebee and EXLA as dependencies in your `mix.exs`. EXLA is an optional dependency but an important one as it allows you to compile models just-in-time and run them on CPU/GPU:

```elixir
def deps do
  [
    {:bumblebee, "~> 0.4.0"},
    {:exla, ">= 0.0.0"}
  ]
end
```

Then configure `Nx` to use EXLA backend by default in your `config/config.exs` file:

```elixir
import Config

config :nx, default_backend: EXLA.Backend
```

To use GPUs, you must [set the `XLA_TARGET` environment variable accordingly](https://github.com/elixir-nx/xla#usage).

In notebooks and scripts, use the following `Mix.install/2` call to both install and configure dependencies:

```elixir
Mix.install(
  [
    {:bumblebee, "~> 0.4.0"},
    {:exla, ">= 0.0.0"}
  ],
  config: [nx: [default_backend: EXLA.Backend]]
)
```

## Usage

To get a sense of what Bumblebee does, look at this example:

```elixir
{:ok, model_info} = Bumblebee.load_model({:hf, "bert-base-uncased"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

serving = Bumblebee.Text.fill_mask(model_info, tokenizer)
Nx.Serving.run(serving, "The capital of [MASK] is Paris.")
#=> %{
#=>   predictions: [
#=>     %{score: 0.9279842972755432, token: "france"},
#=>     %{score: 0.008412551134824753, token: "brittany"},
#=>     %{score: 0.007433671969920397, token: "algeria"},
#=>     %{score: 0.004957548808306456, token: "department"},
#=>     %{score: 0.004369721747934818, token: "reunion"}
#=>   ]
#=> }
```

We load the BERT model from Hugging Face Hub, then plug it into an end-to-end pipeline in the form of "serving", finally we use the serving to get our task done. For more details check out [the documentation](https://hexdocs.pm/bumblebee) and the resources below.

## License

    Copyright (c) 2022 Dashbit

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
