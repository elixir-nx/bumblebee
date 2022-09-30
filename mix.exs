defmodule Bumblebee.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :bumblebee,
      version: @version,
      elixir: "~> 1.13",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger, :inets, :ssl]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:axon, "~> 0.2.0-dev", axon_opts()},
      {:tokenizers,
       github: "elixir-nx/tokenizers", ref: "cb98353a19f45ba127870eb36334f635ca297823"},
      # TODO: Comment me when using tokenizers from release
      {:rustler, ">= 0.0.0", optional: true},
      {:exla, github: "elixir-nx/nx", sparse: "exla", override: true},
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
      {:stb_image, "~> 0.5.0", optional: true},
      {:unpickler, github: "dashbitco/unpickler"},
      {:castore, "~> 0.1.0"},
      {:jason, "~> 1.3.0"},
      {:bypass, "~> 2.1", only: :test},
      {:ex_doc, "~> 0.28", only: :dev, runtime: false}
    ]
  end

  defp axon_opts do
    if path = System.get_env("AXON_PATH") do
      [path: path]
    else
      [github: "elixir-nx/axon", branch: "main"]
    end
  end

  defp docs do
    [
      main: "Bumblebee",
      source_url: "https://github.com/elixir-nx/bumblebee",
      source_ref: "v#{@version}",
      groups_for_modules: [
        Models: [
          Bumblebee.Text.Albert,
          Bumblebee.Text.Bart,
          Bumblebee.Text.Bert,
          Bumblebee.Text.Roberta,
          Bumblebee.Vision.ConvNext,
          Bumblebee.Vision.ResNet,
          Bumblebee.Vision.Deit,
          Bumblebee.Vision.Vit
        ],
        Preprocessors: [
          Bumblebee.Text.AlbertTokenizer,
          Bumblebee.Text.BartTokenizer,
          Bumblebee.Text.BertTokenizer,
          Bumblebee.Text.RobertaTokenizer,
          Bumblebee.Vision.ConvNextFeaturizer,
          Bumblebee.Vision.DeitFeaturizer,
          Bumblebee.Vision.VitFeaturizer
        ],
        Interfaces: [
          Bumblebee.ModelSpec,
          Bumblebee.Featurizer,
          Bumblebee.Tokenizer,
          Bumblebee.HuggingFace.Transformers.Config
        ]
      ],
      before_closing_body_tag: &before_closing_body_tag/1
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
