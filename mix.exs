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
      {:tokenizers, github: "elixir-nx/tokenizers"},
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
          Bumblebee.Text.Bert,
          Bumblebee.Vision.ConvNext,
          Bumblebee.Vision.ResNet,
          Bumblebee.Vision.Vit
        ],
        Preprocessors: [
          Bumblebee.Text.BertTokenizer,
          Bumblebee.Vision.ConvNextFeaturizer
        ],
        Interfaces: [
          Bumblebee.ModelSpec,
          Bumblebee.Featurizer,
          Bumblebee.Tokenizer,
          Bumblebee.HuggingFace.Transformers.Config
        ]
      ]
    ]
  end
end
