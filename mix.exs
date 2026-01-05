defmodule Bumblebee.MixProject do
  use Mix.Project

  @version "0.6.3"
  @description "Pre-trained and transformer Neural Network models in Axon"

  def project do
    [
      app: :bumblebee,
      version: @version,
      description: @description,
      name: "Bumblebee",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      package: package()
    ]
  end

  def application do
    [
      mod: {Bumblebee.Application, []},
      extra_applications: [:logger, :inets, :ssl]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:axon, "~> 0.7.0"},
      # {:axon, github: "elixir-nx/axon", override: true},
      {:tokenizers, "~> 0.4"},
      {:nx, "~> 0.9.0 or ~> 0.10.0"},
      {:exla, ">= 0.0.0", only: [:dev, :test]},
      {:torchx, ">= 0.0.0", only: [:dev, :test]},
      # {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
      # {:exla, github: "elixir-nx/nx", sparse: "exla", override: true, only: [:dev, :test]},
      # {:torchx, github: "elixir-nx/nx", sparse: "torchx", override: true, only: [:dev, :test]},
      {:nx_image, "~> 0.1.0"},
      {:unpickler, "~> 0.1.0"},
      {:safetensors, "~> 0.1.3"},
      {:jason, "~> 1.4.0"},
      {:unzip, "~> 0.12.0 or ~> 0.13.0"},
      {:progress_bar, "~> 3.0"},
      {:stb_image, "~> 0.6.0", only: :test},
      {:bypass, "~> 2.1", only: :test},
      {:ex_doc, "~> 0.28", only: :dev, runtime: false},
      {:nx_signal, "~> 0.2.0"}
    ]
  end

  defp docs do
    [
      main: "Bumblebee",
      source_url: "https://github.com/elixir-nx/bumblebee",
      source_ref: "v#{@version}",
      extras: [
        "notebooks/examples.livemd",
        "notebooks/stable_diffusion.livemd",
        "notebooks/llms.livemd",
        "notebooks/llms_rag.livemd",
        "notebooks/qwen3.livemd",
        "notebooks/function_calling.livemd",
        "notebooks/fine_tuning.livemd",
        "examples/phoenix/README.md"
      ],
      extra_section: "GUIDES",
      groups_for_modules: [
        Tasks: [
          Bumblebee.Audio,
          Bumblebee.Text,
          Bumblebee.Vision,
          Bumblebee.Diffusion.StableDiffusion,
          Bumblebee.Diffusion.StableDiffusionControlNet
        ],
        Models: [
          Bumblebee.Audio.Whisper,
          Bumblebee.Diffusion.ControlNet,
          Bumblebee.Diffusion.StableDiffusion.SafetyChecker,
          Bumblebee.Diffusion.UNet2DConditional,
          Bumblebee.Diffusion.VaeKl,
          Bumblebee.Multimodal.Blip,
          Bumblebee.Multimodal.Clip,
          Bumblebee.Multimodal.LayoutLm,
          Bumblebee.Text.Albert,
          Bumblebee.Text.Bart,
          Bumblebee.Text.Bert,
          Bumblebee.Text.Blenderbot,
          Bumblebee.Text.BlipText,
          Bumblebee.Text.ClipText,
          Bumblebee.Text.Distilbert,
          Bumblebee.Text.Gemma,
          Bumblebee.Text.Gpt2,
          Bumblebee.Text.GptBigCode,
          Bumblebee.Text.GptNeoX,
          Bumblebee.Text.Llama,
          Bumblebee.Text.M2m100,
          Bumblebee.Text.Mbart,
          Bumblebee.Text.Mistral,
          Bumblebee.Text.MpNet,
          Bumblebee.Text.Phi,
          Bumblebee.Text.Phi3,
          Bumblebee.Text.Roberta,
          Bumblebee.Text.SmolLM3,
          Bumblebee.Text.T5,
          Bumblebee.Vision.BlipVision,
          Bumblebee.Vision.ClipVision,
          Bumblebee.Vision.ConvNext,
          Bumblebee.Vision.Deit,
          Bumblebee.Vision.DinoV2,
          Bumblebee.Vision.ResNet,
          Bumblebee.Vision.Swin,
          Bumblebee.Vision.Vit
        ],
        Preprocessors: [
          Bumblebee.Audio.WhisperFeaturizer,
          Bumblebee.Text.PreTrainedTokenizer,
          Bumblebee.Vision.BitFeaturizer,
          Bumblebee.Vision.BlipFeaturizer,
          Bumblebee.Vision.ClipFeaturizer,
          Bumblebee.Vision.ConvNextFeaturizer,
          Bumblebee.Vision.DeitFeaturizer,
          Bumblebee.Vision.VitFeaturizer
        ],
        Schedulers: [
          Bumblebee.Diffusion.DdimScheduler,
          Bumblebee.Diffusion.LcmScheduler,
          Bumblebee.Diffusion.PndmScheduler
        ],
        Interfaces: [
          Bumblebee.Configurable,
          Bumblebee.ModelSpec,
          Bumblebee.Featurizer,
          Bumblebee.Tokenizer,
          Bumblebee.Scheduler,
          Bumblebee.Text.Generation
        ],
        Other: [
          Bumblebee.Text.GenerationConfig,
          Bumblebee.Text.WhisperGenerationConfig
        ]
      ],
      groups_for_docs: [
        # Bumblebee
        Models: &(&1[:type] == :model),
        Featurizers: &(&1[:type] == :featurizer),
        Tokenizers: &(&1[:type] == :tokenizer),
        Schedulers: &(&1[:type] == :scheduler)
      ],
      before_closing_body_tag: &before_closing_body_tag/1
    ]
  end

  def package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => "https://github.com/elixir-nx/bumblebee"
      }
    ]
  end

  # Add KaTeX integration for rendering math
  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css" integrity="sha384-t5CR+zwDAROtph0PXGte6ia8heboACF9R5l/DiY+WZ3P2lxNgvJkQk5n7GPvLMYw" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js" integrity="sha384-FaFLTlohFghEIZkw6VGwmf9ISTubWAVYW8tG8+w2LAIftJEULZABrF9PPFv+tVkH" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/contrib/auto-render.min.js" integrity="sha384-bHBqxz8fokvgoJ/sc17HODNxa42TlaEhB+w8ZJXTc2nZf1VgEaFZeZvT4Mznfz0v" crossorigin="anonymous"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ]
        });
      });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
