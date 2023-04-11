# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
