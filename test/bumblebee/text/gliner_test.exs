defmodule Bumblebee.Text.GlinerTest do
  use ExUnit.Case, async: true
  import Nx.Testing
  import ExUnit.CaptureLog

  for variant <- ["local", "global", "no_refinement"] do
    @variant variant
    test "boundary and span scores match Python for #{@variant}" do
      directory = Path.expand("../../fixtures/gliner/#{@variant}", __DIR__)
      reference = directory |> Path.join("expected.json") |> File.read!() |> Jason.decode!()

      log =
        capture_log(fn ->
          {:ok, info} =
            Bumblebee.load_model({:local, directory},
              spec_overrides: [hidden_size: 12],
              log_params_diff: true
            )

          assert info.spec.architecture == :boundary
          {_, predict} = Axon.build(info.model, compiler: EXLA)
          inputs = Map.new(reference["inputs"], fn {key, value} -> {key, Nx.tensor(value)} end)
          output = predict.(info.params, inputs)

          for {key, expected} <- reference["outputs"] do
            assert_all_close(output[key], Nx.tensor(expected), atol: 2.0e-5, rtol: 2.0e-5)
          end

          {:ok, info} =
            Bumblebee.load_model({:local, directory},
              architecture: :span_scoring,
              spec_overrides: [hidden_size: 12],
              log_params_diff: true
            )

          {_, predict} = Axon.build(info.model, compiler: EXLA)

          inputs =
            Map.new(reference["scorer_inputs"], fn {key, value} -> {key, Nx.tensor(value)} end)

          assert_all_close(predict.(info.params, inputs), Nx.tensor(reference["scores"]),
            atol: 2.0e-5,
            rtol: 2.0e-5
          )
        end)

      refute log =~ "were missing"
      refute log =~ "non-matching shape"
    end
  end

  test "loads a nested DeBERTa encoder from GLiNER parameter names" do
    directory = Path.expand("../../fixtures/gliner/local", __DIR__)
    {:ok, encoder_spec} = Bumblebee.load_spec({:local, Path.join(directory, "encoder_config")})

    {:ok, info} =
      Bumblebee.load_model({:local, directory},
        architecture: :encoder,
        spec_overrides: [encoder_spec: encoder_spec]
      )

    reference = directory |> Path.join("encoder_expected.json") |> File.read!() |> Jason.decode!()
    inputs = Map.new(reference["inputs"], fn {key, value} -> {key, Nx.tensor(value)} end)
    {_, predict} = Axon.build(info.model, compiler: EXLA)

    assert_all_close(
      predict.(info.params, inputs).hidden_state,
      Nx.tensor(reference["hidden_state"]),
      atol: 2.0e-5,
      rtol: 2.0e-5
    )
  end

  test "rejects invalid scoring and refinement options" do
    for options <- [
          [pair_temperature: 0],
          [boundary_ffn_multiplier: -1],
          [abstention_threshold: 2]
        ] do
      assert_raise ArgumentError, fn -> Bumblebee.configure(Bumblebee.Text.Gliner, options) end
    end
  end

  test "rejects unsupported variants before loading weights" do
    for options <- [
          [candidate_pool: "per_query"],
          [query_attention_layers: 1],
          [adaptive_threshold: true],
          [content_soft_max_pool: true]
        ] do
      assert_raise ArgumentError, ~r/unsupported GLiNER option/, fn ->
        Bumblebee.configure(Bumblebee.Text.Gliner, options)
      end
    end
  end
end
