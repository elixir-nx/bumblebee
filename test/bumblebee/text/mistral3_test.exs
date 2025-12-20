defmodule Bumblebee.Text.Mistral3Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:local, "test/fixtures/models/tiny-random-Mistral3Model"})

    assert %Bumblebee.Text.Mistral3{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    # Expected values from Bumblebee (see test/fixtures/scripts/bumblebee_expected_values.txt)
    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [0.7017732858657837, 0.5815300941467285, -0.9297741055488586],
          [-2.16787052154541, -0.01968071237206459, -1.0697519779205322],
          [-1.0169540643692017, 0.6504985094070435, -1.6784638166427612]
        ]
      ]),
      atol: 1.0e-4
    )
  end

  test ":base with interleaved attention" do
    assert {:ok, spec} =
             Bumblebee.load_spec({:local, "test/fixtures/models/tiny-random-Mistral3Model"})

    # Verify interleaved attention is enabled by default
    assert spec.use_interleaved_attention == true

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:local, "test/fixtures/models/tiny-random-Mistral3Model"},
               spec: spec
             )

    assert %Bumblebee.Text.Mistral3{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}
  end

  test ":base without interleaved attention" do
    assert {:ok, spec} =
             Bumblebee.load_spec({:local, "test/fixtures/models/tiny-random-Mistral3Model"})

    # Disable interleaved attention to use sliding window on all layers
    spec = Bumblebee.configure(spec, use_interleaved_attention: false)

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:local, "test/fixtures/models/tiny-random-Mistral3Model"},
               spec: spec
             )

    assert %Bumblebee.Text.Mistral3{
             architecture: :base,
             use_interleaved_attention: false
           } = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:local, "test/fixtures/models/tiny-random-Mistral3ForSequenceClassification"}
             )

    assert %Bumblebee.Text.Mistral3{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    # Expected values from Bumblebee (see test/fixtures/scripts/bumblebee_expected_values.txt)
    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.08115436881780624, -0.045208640396595]]),
      atol: 1.0e-4
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:local, "test/fixtures/models/tiny-random-Mistral3ForCausalLM"}
             )

    assert %Bumblebee.Text.Mistral3{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    # vocab_size is 1024 in the tiny-random model
    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    # Expected values from Bumblebee (see test/fixtures/scripts/bumblebee_expected_values.txt)
    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [-0.061699170619249344, -0.004930073395371437, 0.16922777891159058],
          [-0.055778875946998596, 0.07242244482040405, -0.020687159150838852],
          [0.12626346945762634, 0.09094549715518951, 0.21130035817623138]
        ]
      ]),
      atol: 1.0e-4
    )
  end

  # Test that module structure and options are correct
  test "module structure" do
    assert Bumblebee.Text.Mistral3.architectures() == [
             :base,
             :for_causal_language_modeling,
             :for_sequence_classification
           ]

    # Test default configuration
    spec = %Bumblebee.Text.Mistral3{}
    assert spec.architecture == :base
    assert spec.vocab_size == 131_072
    assert spec.max_positions == 262_144
    assert spec.hidden_size == 4096
    assert spec.intermediate_size == 14336
    assert spec.num_blocks == 32
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 8
    assert spec.attention_window_size == 4096
    assert spec.use_interleaved_attention == true
    assert spec.activation == :silu
    assert spec.layer_norm_epsilon == 1.0e-5
    assert spec.rotary_embedding_base == 1_000_000
  end

  test "configuration" do
    spec = %Bumblebee.Text.Mistral3{}

    configured =
      Bumblebee.Text.Mistral3.config(spec,
        vocab_size: 65536,
        use_interleaved_attention: false
      )

    assert configured.vocab_size == 65536
    assert configured.use_interleaved_attention == false
  end
end
