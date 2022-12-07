# Bumblebee

Bumblebee provides pre-trained and transformer Neural Network models on top of Axon. It includes integration with [🤗 Models](https://huggingface.co/models), allowing anyone to download and perform Machine Learning tasks with few lines of code.

![Numbat and Bumblebees](.github/images/background.jpg)

## Usage

First add Bumblebee and EXLA as dependencies. EXLA is an optional dependency but an important one as it allows you to compile models just-in time and run them on CPU/GPU:

```elixir
def deps do
  [
    {:bumblebee, "~> 0.1.0"},
    {:exla, ">= 0.0.0"}
  ]
end
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
