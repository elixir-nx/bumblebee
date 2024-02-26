# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.5.3](https://github.com/elixir-nx/bumblebee/tree/v0.5.3) (2024-02-26)

### Fixed

* Loading generation config with nil attributes
* Generating with `:no_repeat_ngram_length` when using lower precision

## [v0.5.2](https://github.com/elixir-nx/bumblebee/tree/v0.5.2) (2024-02-24)

### Fixed

* Fixed loading sharded parameters in the safetensors format

## [v0.5.1](https://github.com/elixir-nx/bumblebee/tree/v0.5.1) (2024-02-23)

### Fixed

* Fixed loading Mistral configuration with attention window disabled

## [v0.5.0](https://github.com/elixir-nx/bumblebee/tree/v0.5.0) (2024-02-23)

This release changes the directory structure of the models cache, such that cached files from the same HuggingFace Hub repository are grouped in a separate subdirectory. This change is meant to simplify the process of manually removing specific models from the cache to free up space. As a result, the cache contents from prior versions are invalidated, so you most likely want to remove the current cache contents. To find the cache location run `elixir -e 'Mix.install([{:bumblebee, "0.4.2"}]); IO.puts(Bumblebee.cache_dir())'` (defaults to the standard cache location for the given operating system).

We also reduced memory usage during parameter loading (both when loading onto the CPU and GPU directly). Previously, larger models sometimes required loading parameters using CPU and only then transfering to the GPU, in order to avoid running out of GPU memory during parameter transformations. With this release this should no longer be the case. Loading parameters now has barely any memory footprint other than the parameters themselves.

### Added

* Notebook on LLaMA 2 to the docs ([#259](https://github.com/elixir-nx/bumblebee/pull/259))
* Mistral model ([#264](https://github.com/elixir-nx/bumblebee/pull/264))
* Projection head models for ClipText and ClipVision ([#276](https://github.com/elixir-nx/bumblebee/pull/276))
* Support more rotary embedding options for LLaMA required for Deepseek Coder ([#285](https://github.com/elixir-nx/bumblebee/pull/285))
* Temperature generation option ([#290](https://github.com/elixir-nx/bumblebee/pull/290))
* GPTBigCode model (used by Starcoder) ([#294](https://github.com/elixir-nx/bumblebee/pull/294))
* Automatic detection of diffusers params files (specifying `:params_filename` for Stable Diffusion models is no longer necessary) ([#301](https://github.com/elixir-nx/bumblebee/pull/301))
* `:seed` option to generation serving inputs ([#303](https://github.com/elixir-nx/bumblebee/pull/303))
* `:params_variant` option to `Bumblebee.load_model/2` for loading parameters of different precision ([#309](https://github.com/elixir-nx/bumblebee/pull/309))
* `:type` option to `Bumblebee.load_model/2` for loading model under a specific precision policy ([#311](https://github.com/elixir-nx/bumblebee/pull/311))
* LCM scheduler ([#320](https://github.com/elixir-nx/bumblebee/pull/320))
* Token summary to text generation output ([#336](https://github.com/elixir-nx/bumblebee/pull/336))
* DINOv2 model ([#334](https://github.com/elixir-nx/bumblebee/pull/334))
* `:spec_overrides` option to `Bumblebee.load_model/2` ([#340](https://github.com/elixir-nx/bumblebee/pull/340))
* Support for attention sliding window in Mistral ([#341](https://github.com/elixir-nx/bumblebee/pull/341))

### Changed

* **(Breaking)** Text generation to always return only the new text (for some models it used to include the prompt) ([#302](https://github.com/elixir-nx/bumblebee/pull/302))
* Deprecated all options in `Bumblebee.apply_tokenizer/3`, these should now be set on the tokenizer using `Bumblebee.configure/2` ([#310](https://github.com/elixir-nx/bumblebee/pull/310))
* Reduced memory used when the `:preallocate_params` serving option is enabled ([#317](https://github.com/elixir-nx/bumblebee/pull/317))
* **(Breaking)** Changed image size to maps in image featurizers ([#329](https://github.com/elixir-nx/bumblebee/pull/329))
* **(Breaking)** Renamed ViT and DeiT `:for_masked_image_modeling` output from `:logits` to `:pixel_values`
* **(Breaking)** Renamed CLIP outputs `:text_embeddings` and `:image_embeddings` to singular
* **(Breaking)** Changed ResNet `:pooled_state` output to flatten the extra 1-sized axes
* Cache directory structure to group files by repository ([#332](https://github.com/elixir-nx/bumblebee/pull/332))
* **(Breaking)** Changed the output of `Bumblebee.Text.Generation.build_generate/4` to a map ([#336](https://github.com/elixir-nx/bumblebee/pull/336))
* Reduced memory usage during parameter loading ([#344](https://github.com/elixir-nx/bumblebee/pull/344))

### Removed

* Removed the serving `:seed` option in favour of a runtime, per-input seed ([#303](https://github.com/elixir-nx/bumblebee/pull/303))
* Conversational serving ([#308](https://github.com/elixir-nx/bumblebee/pull/308))
* Specific tokenizer modules in favour of a single module ([#310](https://github.com/elixir-nx/bumblebee/pull/310))
* Removed the deprecated `Bumblebee.Audio.speech_to_text/5` (in favour of the more specific `speech_to_text_whisper/5`)

### Fixed

* Featurizer batch template when image size is a tuple
* Error in concatenating results when running servings as partitioned ([#282](https://github.com/elixir-nx/bumblebee/pull/282))
* Decoder cache being casted with low precision policies ([#299](https://github.com/elixir-nx/bumblebee/pull/299))
* Loading of more recent VAE KL checkpoints ([#305](https://github.com/elixir-nx/bumblebee/pull/305))
* Tokenizers truncation to account for trailing special tokens ([#307](https://github.com/elixir-nx/bumblebee/pull/307))
* Loading models with auth token from within a HuggingFace Space ([#314](https://github.com/elixir-nx/bumblebee/pull/314))
* Zero-shot classification serving to handle uppercased entailment token in model config ([#327](https://github.com/elixir-nx/bumblebee/pull/327))
* Fixed text generation when using lower precision and encoder-decoder models (such as Whisper) ([#346](https://github.com/elixir-nx/bumblebee/pull/346))

## [v0.4.2](https://github.com/elixir-nx/bumblebee/tree/v0.4.2) (2023-09-28)

### Added

* More detailed error messages when loading fails ([#256](https://github.com/elixir-nx/bumblebee/pull/256))

### Changed

* Automatic detection there are no model parameters in the `.bin` format, but `.safetensors` is available ([#256](https://github.com/elixir-nx/bumblebee/pull/256))

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
{:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
+{:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

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
