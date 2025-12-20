defmodule Bumblebee.Text.MistralTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MistralModel"})

    assert %Bumblebee.Text.Mistral{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.9450, -1.3945, 0.7331], [-2.1118, -1.3091, -0.7834], [-1.7609, -1.3034, 1.0634]]
      ])
    )
  end

  test ":base with attention sliding window" do
    assert {:ok, spec} =
             Bumblebee.load_spec({:hf, "hf-internal-testing/tiny-random-MistralModel"})

    spec = Bumblebee.configure(spec, attention_window_size: 2)

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MistralModel"},
               spec: spec
             )

    assert %Bumblebee.Text.Mistral{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.9450, -1.3945, 0.7331], [-2.1118, -1.3091, -0.7834], [-1.3033, -1.3374, 0.8919]]
      ])
    )
  end

  test ":base with interleaved attention" do
    assert {:ok, spec} =
             Bumblebee.load_spec({:hf, "hf-internal-testing/tiny-random-MistralModel"})

    # Enable interleaved attention: even layers use global, odd layers use sliding window
    spec = Bumblebee.configure(spec, attention_window_size: 2, use_interleaved_attention: true)

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MistralModel"},
               spec: spec
             )

    assert %Bumblebee.Text.Mistral{architecture: :base, use_interleaved_attention: true} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    # With interleaved attention, even layers (0, 2, 4...) use global attention
    # and odd layers (1, 3, 5...) use sliding window attention
    # The output should be different from both pure global and pure sliding window
    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.9450, -1.3945, 0.7331], [-2.1118, -1.3091, -0.7834], [-1.4057, -1.2495, 0.8730]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-MistralForSequenceClassification"}
             )

    assert %Bumblebee.Text.Mistral{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[0.0035, -0.0357]])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MistralForCausalLM"})

    assert %Bumblebee.Text.Mistral{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 32000}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.1054, 0.0026, 0.0450], [0.1400, 0.1388, 0.0265], [0.0060, -0.1150, -0.1463]]
      ])
    )
  end
end
