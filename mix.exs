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

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :inets, :ssl]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.1.0-dev", axon_opts()},
      {:exla, github: "elixir-nx/nx", sparse: "exla", override: true},
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
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
          Bumblebee.Vision.ResNet
        ],
        Interfaces: [
          Bumblebee.ModelSpec,
          Bumblebee.HuggingFace.Transformers.Config
        ]
      ]
    ]
  end
end
