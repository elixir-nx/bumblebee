defmodule Bumblebee.Text.EmbeddingParityTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @fixtures Path.expand("../../fixtures/embedding_models", __DIR__)
  # Float32 CPU reductions differ slightly between PyTorch and XLA. 1e-4
  # leaves room for rounding while detecting mask and RoPE formula errors.
  @models ~w(gemma3_text llama mistral qwen3 qwen3_linear qwen3_dynamic)

  for name <- @models, mode <- ~w(causal bidirectional) do
    @name name
    @mode mode

    test "#{name} #{@mode} matches Transformers hidden states" do
      {model, params, reference} = load_fixture(@name, @mode)

      outputs =
        Axon.predict(model, params, %{
          "input_ids" => Nx.tensor(reference["input_ids"]),
          "attention_mask" => Nx.tensor(reference["attention_mask"])
        })

      assert_all_close(outputs.hidden_state, Nx.tensor(reference["hidden_state"]), atol: 1.0e-4)

      changed_outputs =
        Axon.predict(model, params, %{
          "input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 9]]),
          "attention_mask" => Nx.tensor(reference["attention_mask"])
        })

      assert_all_close(
        changed_outputs.hidden_state,
        Nx.tensor(reference["changed_future_hidden_state"]),
        atol: 1.0e-4
      )

      padded_outputs =
        Axon.predict(model, params, %{
          "input_ids" => Nx.tensor(reference["padded_input_ids"]),
          "attention_mask" => Nx.tensor(reference["padded_attention_mask"]),
          "position_ids" => Nx.tensor(reference["padded_position_ids"])
        })

      assert_all_close(
        padded_outputs.hidden_state,
        Nx.tensor(reference["padded_hidden_state"]),
        atol: 1.0e-4
      )

      compact_outputs =
        Axon.predict(model, params, %{
          "input_ids" => Nx.tensor([[1, 2, 3, 4]]),
          "attention_mask" => Nx.tensor([[1, 1, 1, 1]])
        })

      assert_all_close(compact_outputs.hidden_state, Nx.tensor(reference["compact_hidden_state"]),
        atol: 1.0e-4,
        rtol: 1.0e-4
      )

      assert_all_close(
        padded_outputs.hidden_state[[.., 0..3, ..]],
        compact_outputs.hidden_state,
        atol: 1.0e-4
      )

      first_token_changed? =
        outputs.hidden_state[[.., 0, ..]]
        |> Nx.all_close(
          changed_outputs.hidden_state[[.., 0, ..]],
          atol: 1.0e-6,
          rtol: 1.0e-6
        )
        |> Nx.to_number() == 0

      assert first_token_changed? == (@mode == "bidirectional")
    end
  end

  defp load_fixture(name, mode) do
    path = Path.join(@fixtures, "tiny-random-#{name}-#{mode}")
    reference = path |> Path.join("reference.json") |> File.read!() |> Jason.decode!()

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:local, path})

    assert reference["transformers_version"] == "5.19.0.dev0"
    assert reference["transformers_commit"] == "3693f8d26311305e914735a6373fb03468d6aaa0"
    assert spec.use_bidirectional_attention == (mode == "bidirectional")

    {model, params, reference}
  end
end
