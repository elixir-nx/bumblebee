# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.4.1](https://github.com/elixir-nx/bumblebee/tree/v0.4.1) (2023-09-25)

### Changed

* Aligned batch handling for serving run and batched run ([#252](https://github.com/elixir-nx/bumblebee/pull/252))

### Fixed

* Fixed `:top_k` upper bound in classification servings

## [v0.4.0](https://github.com/elixir-nx/bumblebee/tree/v0.4.0) (2023-09-14)

### Added

* Chunking options to speech-to-text to support long audio transcription ([#236](https://github.com/elixir-nx/bumblebee/pull/236))
* Support for Whisper timestamps and task/language configuration ([#238](https://github.com/elixir-nx/bumblebee/pull/238))
* Support for streaming speech-to-text results ([#242](https://github.com/elixir-nx/bumblebee/pull/242))
* Introduced featurizer batch phase that is compiled as part of the serving computation ([#243](https://github.com/elixir-nx/bumblebee/pull/243))
* Removed possibly contentious Nx calls from serving postprocessing ([#244](https://github.com/elixir-nx/bumblebee/pull/244), [#245](https://github.com/elixir-nx/bumblebee/pull/245))

### Changed

* Deprecated `Bumblebee.Audio.speech_to_text/5` in favour of the more specific `Bumblebee.Audio.speech_to_text_whisper/5`
* Changed the tensors returned from embedding servings to use `Nx.BinaryBackend`

## [v0.3.1](https://github.com/elixir-nx/bumblebee/tree/v0.3.1) (2023-08-17)

### Added

* LLaMA model ([#199](https://github.com/elixir-nx/bumblebee/pull/199))
* GPT-NeoX model ([#204](https://github.com/elixir-nx/bumblebee/pull/204))
* Option to customize scores function in classification tasks ([#211](https://github.com/elixir-nx/bumblebee/pull/211))
* Text embedding serving ([#214](https://github.com/elixir-nx/bumblebee/pull/214))
* `Bumblebee.cache_dir/0` for discovering cache location ([#220](https://github.com/elixir-nx/bumblebee/pull/220))
* Image embedding serving ([#229](https://github.com/elixir-nx/bumblebee/pull/229))
* Support for compiling text servings for multiple sequence lengths ([#228](https://github.com/elixir-nx/bumblebee/pull/228))
* Support for streaming chunks during text generation ([#232](https://github.com/elixir-nx/bumblebee/pull/232))
* Added `:preallocate_params` option to all servings, useful with multiple GPUs ([#233](https://github.com/elixir-nx/bumblebee/pull/233))
* Support for loading params in the .safetensors format ([#231](https://github.com/elixir-nx/bumblebee/pull/231))

## [v0.3.0](https://github.com/elixir-nx/bumblebee/tree/v0.3.0) (2023-04-14)

In this release we moved all generation options to a new `%Bumblebee.Text.GenerationConfig{}` struct, which needs to be explicitly loaded and configured. A number of generation options is model-specific and they used to be a part of model specification, but encapsulating everything in a single struct improves the transparency of options origin and reconfiguration. The text generation servings (generation, speech-to-text and conversation) need to be adjusted as follows:

```diff
{:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})
+{:ok, generation_config} = Bumblebee.load_generation_config({:hf, "gpt2"})

+generation_config = Bumblebee.configure(generation_config, max_new_tokens: 100)
+serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)
-serving = Bumblebee.Text.generation(model_info, tokenizer, max_new_tokens: 100)
```

### Added

* Word-based aggregations for token classification ([#174](https://github.com/elixir-nx/bumblebee/pull/174))
* BLIP model ([#181](https://github.com/elixir-nx/bumblebee/pull/181))
* Text-to-image serving ([#181](https://github.com/elixir-nx/bumblebee/pull/181))
* Generation option to avoid repeated n-grams ([#182](https://github.com/elixir-nx/bumblebee/pull/182))
* Blenderbot model ([#177](https://github.com/elixir-nx/bumblebee/pull/177))
* Option to load models from cache without outgoing traffic ([#183](https://github.com/elixir-nx/bumblebee/pull/183))
* Whisper Phoenix demo ([#184](https://github.com/elixir-nx/bumblebee/pull/184))
* Image channels normalization in featurizers ([#189](https://github.com/elixir-nx/bumblebee/pull/189))
* T5 encoder model ([#190](https://github.com/elixir-nx/bumblebee/pull/190))
* Contrastive search for sequence generation ([#192](https://github.com/elixir-nx/bumblebee/pull/192))
* Multinomial sampling for sequence generation ([#161](https://github.com/elixir-nx/bumblebee/pull/161))
* Support for loading sharded params checkpoints ([#200](https://github.com/elixir-nx/bumblebee/pull/200))

### Changed

* Model loading to not log params diff if everything is loaded correctly ([#186](https://github.com/elixir-nx/bumblebee/pull/186))
* Moved all generation options to a new `%Bumblebee.Text.GenerationConfig{}` struct ([#193](https://github.com/elixir-nx/bumblebee/pull/193))

## [v0.2.0](https://github.com/elixir-nx/bumblebee/tree/v0.2.0) (2023-03-16)

### Added

* Support for Stable Diffusion v2 ([#117](https://github.com/elixir-nx/bumblebee/pull/117))
* CamemBERT model ([#110](https://github.com/elixir-nx/bumblebee/pull/110))
* XLM-RoBERTa model ([#136](https://github.com/elixir-nx/bumblebee/pull/136))
* Support for configuring backend used by `load_model` ([#140](https://github.com/elixir-nx/bumblebee/pull/140))
* Zero-shot classification serving ([#121](https://github.com/elixir-nx/bumblebee/pull/121), [#145](https://github.com/elixir-nx/bumblebee/pull/145))
* Support parameter files using Zip64 ([#144](https://github.com/elixir-nx/bumblebee/pull/144))
* Notebook exemplifying fine-tuning with Axon ([#102](https://github.com/elixir-nx/bumblebee/pull/102))
* Whisper model ([#107](https://github.com/elixir-nx/bumblebee/pull/107))
* Speech-to-text serving ([#107](https://github.com/elixir-nx/bumblebee/pull/107))
* Support for loading tokenizers with overridden special tokens ([#141](https://github.com/elixir-nx/bumblebee/pull/141))
* Question answering serving ([#157](https://github.com/elixir-nx/bumblebee/pull/157))
* T5 model ([#159](https://github.com/elixir-nx/bumblebee/pull/159))
* Conversational serving ([#165](https://github.com/elixir-nx/bumblebee/pull/165))
* DistilBERT model ([#172](https://github.com/elixir-nx/bumblebee/pull/172))

### Changed

* Cancel download request when the caller terminates ([#135](https://github.com/elixir-nx/bumblebee/pull/135))
* Introduced explicit parameter mapping, which changed parameter names in all models ([#148](https://github.com/elixir-nx/bumblebee/pull/148))

### Fixed

* Initialization of parameters with mismatched shape

## [v0.1.2](https://github.com/elixir-nx/bumblebee/tree/v0.1.2) (2022-12-15)

### Fixed

* Stable Diffusion to accept a string as the input ([#115](https://github.com/elixir-nx/bumblebee/pull/115))

## [v0.1.1](https://github.com/elixir-nx/bumblebee/tree/v0.1.1) (2022-12-14)

### Added

* Support for a negative prompt in Stable Diffusion input ([#109](https://github.com/elixir-nx/bumblebee/pull/109))

### Changed

* Improved fill-mask output token to not include whitespace or whitespace placeholder ([#106](https://github.com/elixir-nx/bumblebee/pull/106))

### Fixed

* Image rendering in the Stable Diffusion notebook ([#95](https://github.com/elixir-nx/bumblebee/pull/95))
* Download error when the tmp and cache directories are on different file systems ([#98](https://github.com/elixir-nx/bumblebee/pull/98))

## [v0.1.0](https://github.com/elixir-nx/bumblebee/tree/v0.1.0) (2022-12-07)

Initial release.
