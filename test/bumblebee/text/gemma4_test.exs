defmodule Bumblebee.Text.Gemma4Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Gemma4TextModel"})

    assert %Bumblebee.Text.Gemma4{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-1.9693, -0.4030, -0.3734], [-0.3044, 0.2130, -0.3618], [0.1315, -0.3031, 0.2566]]
      ]),
      atol: 1.0e-4
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Gemma4ForCausalLM"})

    assert %Bumblebee.Text.Gemma4{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0629, 0.2883, 0.1995], [-2.2633, -0.3478, -2.4386], [-0.9260, -0.6100, -1.8994]]
      ]),
      atol: 1.0e-4
    )
  end

  test "uses token IDs for per-layer inputs with supplied embeddings" do
    {:ok, %{model: model, params: params, spec: spec}} =
      Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Gemma4ForCausalLM"})

    ids = Nx.tensor([[10, 20, 30]])

    embeddings =
      params.data["embedder.token_embedding"]["kernel"]
      |> Nx.take(ids)
      |> Nx.multiply(:math.sqrt(spec.hidden_size))

    from_ids = Axon.predict(model, params, %{"input_ids" => ids})

    from_both =
      Axon.predict(model, params, %{"input_ids" => ids, "input_embeddings" => embeddings})

    from_embeddings = Axon.predict(model, params, %{"input_embeddings" => embeddings})

    assert_all_close(from_both.logits, from_ids.logits)
    assert Nx.to_number(Nx.all_close(from_both.logits, from_embeddings.logits)) == 0
  end

  test ":for_causal_language_modeling from a multimodal checkpoint" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Gemma4ForConditionalGeneration"}
             )

    assert %Bumblebee.Text.Gemma4{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-1.8881, -1.2950, 1.3104], [-0.0950, -0.6332, 0.2414], [-0.0770, 1.0486, 1.4560]]
      ]),
      atol: 1.0e-4
    )
  end

  test ":for_causal_language_modeling with keys as values" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true"}
             )

    assert %Bumblebee.Text.Gemma4{
             architecture: :for_causal_language_modeling,
             attention_key_equals_value: true,
             global_num_key_value_heads: 1,
             per_layer_input_size: 0
           } = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.6174, -0.4704, 2.0769], [0.1023, -0.8646, 0.9903], [-0.4417, -0.5268, -0.6976]]
      ]),
      atol: 1.0e-4
    )
  end

  test "loads untied output weights from text-only and multimodal checkpoints" do
    head = Nx.broadcast(0.25, {1024, 32})

    for checkpoint <- [
          "tiny-random-Gemma4ForCausalLM",
          "tiny-random-Gemma4ForConditionalGeneration"
        ] do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "bumblebee-testing/" <> checkpoint},
                 spec_overrides: [tie_word_embeddings: false],
                 safetensors_reader: fn path ->
                   path
                   |> Safetensors.read!(lazy: true)
                   |> Map.put("lm_head.weight", head)
                 end
               )

      assert_equal(params.data["language_modeling_head.output"]["kernel"], head)
    end
  end

  test "uses the multimodal config to decide whether to tie output weights" do
    data = %{
      "model_type" => "gemma4",
      "tie_word_embeddings" => false,
      "text_config" => %{
        "model_type" => "gemma4_text",
        "hidden_size" => 32,
        "global_head_dim" => 16,
        "tie_word_embeddings" => true
      }
    }

    assert %Bumblebee.Text.Gemma4{hidden_size: 32, tie_word_embeddings: false} =
             Bumblebee.HuggingFace.Transformers.Config.load(%Bumblebee.Text.Gemma4{}, data)
  end

  test "generation with shared key-value blocks" do
    repo = {:hf, "bumblebee-testing/tiny-random-Gemma4ForCausalLM"}
    assert {:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model(repo)
    {:ok, generation_config} = Bumblebee.load_generation_config(repo)

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 8,
        strategy: %{type: :greedy_search},
        eos_token_id: nil
      )

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
      "seed" => Nx.tensor([0])
    }

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[409, 492, 232, 458, 388, 376, 492, 492]]))
  end

  test "loads global attention shapes from per-layer configuration" do
    data = %{
      "model_type" => "gemma4_text",
      "num_hidden_layers" => 2,
      "layer_types" => ["sliding_attention", "full_attention"],
      "head_dim" => 8,
      "rope_parameters" => %{
        "sliding_attention" => %{"rope_theta" => 10_000.0},
        "full_attention" => %{"rope_theta" => 1_000_000.0, "partial_rotary_factor" => 0.25}
      },
      "attention_k_eq_v" => true,
      "per_layer_config" => %{
        "00" => %{"head_dim" => 8, "num_key_value_heads" => 2},
        "01" => %{"head_dim" => 16, "num_key_value_heads" => 1}
      }
    }

    spec = Bumblebee.HuggingFace.Transformers.Config.load(%Bumblebee.Text.Gemma4{}, data)

    assert %Bumblebee.Text.Gemma4{
             attention_head_size: 8,
             global_attention_head_size: 16,
             global_num_key_value_heads: 1
           } = spec
  end
end
