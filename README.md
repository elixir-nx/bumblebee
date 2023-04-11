# Bumblebee

[![Docs](https://img.shields.io/badge/hex.pm-docs-8e7ce6.svg)](https://hexdocs.pm/bumblebee)
[![Actions Status](https://github.com/livebook-dev/kino_bumblebee/workflows/Test/badge.svg)](https://github.com/elixir-nx/bumblebee/actions)

Bumblebee provides pre-trained and transformer Neural Network models on top of [Axon](https://github.com/elixir-nx/axon). It includes integration with [🤗 Models](https://huggingface.co/models), allowing anyone to download and perform Machine Learning tasks with few lines of code.

To see all supported architectures, [check out our documentation sidebar](https://hexdocs.pm/bumblebee).

![Numbat and Bumblebees](.github/images/background.jpg)

## Usage

First add Bumblebee and EXLA as dependencies. EXLA is an optional dependency but an important one as it allows you to compile models just-in time and run them on CPU/GPU:

```elixir
def deps do
  [
    {:bumblebee, "~> 0.2.0"},
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
    {:bumblebee, "~> 0.2.0"},
    {:exla, ">= 0.0.0"}
  ],
  config: [nx: [default_backend: EXLA.Backend]]
)
```

## Examples

To explore Bumblebee:

  * See [examples/phoenix](examples/phoenix) for single-file examples of running Neural Networks inside your Phoenix (+ LiveView) apps

  * Use Bumblebee's integration with Livebook v0.8 (or later) to automatically generate "Neural Networks tasks" from the "+ Smart" cell menu (thanks to [`:kino_bumblebee`](https://github.com/livebook-dev/kino_bumblebee))

  * For a more hands on approach, read our example [notebooks](notebooks)

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
