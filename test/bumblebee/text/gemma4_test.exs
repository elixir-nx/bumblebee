defmodule Bumblebee.Text.Gemma4Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Gemma4ForCausalLM"}
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
        [0.1362, -0.0709, 0.1196],
        [0.1083, -0.0170, -0.0931],
        [0.0512, 0.0156, -0.1358]
      ]),
      atol: 1.0e-4
    )
  end

  describe "config" do
    test "default spec has E4B dimensions" do
      spec = Bumblebee.configure(Bumblebee.Text.Gemma4)
      assert spec.architecture == :base
      assert spec.hidden_size == 2560
      assert spec.num_blocks == 42
      assert spec.attention_head_size == 256
      assert spec.global_attention_head_size == 512
    end

    test "loads E4B config from HuggingFace JSON" do
      config = %{
        "model_type" => "gemma4",
        "text_config" => %{
          "hidden_size" => 2560, "intermediate_size" => 10240,
          "num_hidden_layers" => 42, "num_attention_heads" => 8,
          "num_key_value_heads" => 2, "head_dim" => 256,
          "global_head_dim" => 512, "hidden_activation" => "gelu_pytorch_tanh",
          "rms_norm_eps" => 1.0e-6, "vocab_size" => 262_144,
          "max_position_embeddings" => 131_072, "sliding_window" => 512,
          "enable_moe_block" => false, "num_experts" => nil,
          "top_k_experts" => nil, "hidden_size_per_layer_input" => 256,
          "num_kv_shared_layers" => 18, "attention_k_eq_v" => false,
          "tie_word_embeddings" => true, "initializer_range" => 0.02,
          "attention_bias" => false, "final_logit_softcapping" => 30.0,
          "vocab_size_per_layer_input" => 262_144,
          "layer_types" =>
            (List.duplicate("sliding_attention", 5) ++ ["full_attention"])
            |> List.duplicate(7)
            |> List.flatten(),
          "rope_parameters" => %{
            "sliding_attention" => %{"rope_theta" => 10_000.0, "rope_type" => "default"},
            "full_attention" => %{
              "rope_theta" => 1_000_000.0,
              "rope_type" => "proportional",
              "partial_rotary_factor" => 0.25
            }
          }
        }
      }

      spec = %Bumblebee.Text.Gemma4{architecture: :for_causal_language_modeling}
      spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, config)

      assert spec.hidden_size == 2560
      assert spec.num_blocks == 42
      assert spec.enable_moe_block == false
      assert spec.num_kv_shared_layers == 18
      assert spec.rotary_embedding_base_local == 10_000.0
      assert spec.rotary_embedding_base == 1_000_000.0
      assert length(spec.layer_types) == 42
    end

    test "loads 26B MoE config" do
      config = %{
        "model_type" => "gemma4",
        "text_config" => %{
          "hidden_size" => 2816, "intermediate_size" => 2112,
          "num_hidden_layers" => 30, "num_attention_heads" => 16,
          "num_key_value_heads" => 8, "head_dim" => 256,
          "global_head_dim" => 512, "hidden_activation" => "gelu_pytorch_tanh",
          "rms_norm_eps" => 1.0e-6, "vocab_size" => 262_144,
          "max_position_embeddings" => 262_144, "sliding_window" => 1024,
          "enable_moe_block" => true, "num_experts" => 128,
          "top_k_experts" => 8, "moe_intermediate_size" => 704,
          "hidden_size_per_layer_input" => 0,
          "num_kv_shared_layers" => 0, "attention_k_eq_v" => true,
          "num_global_key_value_heads" => 2,
          "tie_word_embeddings" => true, "initializer_range" => 0.02,
          "attention_bias" => false, "final_logit_softcapping" => 30.0,
          "vocab_size_per_layer_input" => 262_144,
          "layer_types" =>
            (List.duplicate("sliding_attention", 5) ++ ["full_attention"])
            |> List.duplicate(5)
            |> List.flatten(),
          "rope_parameters" => %{
            "sliding_attention" => %{"rope_theta" => 10_000.0, "rope_type" => "default"},
            "full_attention" => %{
              "rope_theta" => 1_000_000.0,
              "rope_type" => "proportional",
              "partial_rotary_factor" => 0.25
            }
          }
        }
      }

      spec = %Bumblebee.Text.Gemma4{architecture: :for_causal_language_modeling}
      spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, config)

      assert spec.enable_moe_block == true
      assert spec.num_experts == 128
      assert spec.top_k_experts == 8
      assert spec.attention_k_eq_v == true
    end
  end

  describe "forward pass" do
    test "sliding + full attention" do
      spec = tiny_spec(layer_types: [:sliding_attention, :full_attention])
      model = Bumblebee.build_model(spec)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(%{"input_ids" => Nx.template({1, 8}, :s64)}, Axon.ModelState.empty())
      result = predict_fn.(params, %{"input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])})

      assert Nx.shape(result.logits) == {1, 8, 100}
    end

    test "partial rotary embedding" do
      spec =
        tiny_spec(
          num_blocks: 1,
          layer_types: [:full_attention],
          global_attention_head_size: 64,
          partial_rotary_factor: 0.25
        )

      model = Bumblebee.build_model(spec)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(%{"input_ids" => Nx.template({1, 4}, :s64)}, Axon.ModelState.empty())
      result = predict_fn.(params, %{"input_ids" => Nx.tensor([[1, 2, 3, 4]])})

      assert Nx.shape(result.logits) == {1, 4, 100}
    end

    test "MoE with router + experts" do
      spec =
        tiny_spec(
          enable_moe_block: true,
          num_experts: 4,
          top_k_experts: 2,
          moe_intermediate_size: 8
        )

      model = Bumblebee.build_model(spec)
      {init_fn, predict_fn} = Axon.build(model, compiler: Nx.Defn.Evaluator)
      params = init_fn.(%{"input_ids" => Nx.template({1, 4}, :s64)}, Axon.ModelState.empty())
      result = predict_fn.(params, %{"input_ids" => Nx.tensor([[1, 2, 3, 4]])})

      assert Nx.shape(result.logits) == {1, 4, 100}
    end

    test "with attention mask" do
      spec = tiny_spec()
      model = Bumblebee.build_model(spec)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(%{"input_ids" => Nx.template({1, 8}, :s64)}, Axon.ModelState.empty())

      result =
        predict_fn.(params, %{
          "input_ids" => Nx.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]),
          "attention_mask" => Nx.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
        })

      assert Nx.shape(result.logits) == {1, 8, 100}
    end

    test "logit softcapping bounds output" do
      spec = tiny_spec(final_logit_softcapping: 10.0)
      model = Bumblebee.build_model(spec)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(%{"input_ids" => Nx.template({1, 4}, :s64)}, Axon.ModelState.empty())
      result = predict_fn.(params, %{"input_ids" => Nx.tensor([[1, 2, 3, 4]])})

      max_val = Nx.to_number(Nx.reduce_max(Nx.abs(result.logits)))
      assert max_val <= 10.01
    end
  end

  defp tiny_spec(overrides \\ []) do
    defaults = [
      architecture: :for_causal_language_modeling,
      hidden_size: 64,
      intermediate_size: 128,
      num_blocks: 2,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      attention_head_size: 32,
      global_attention_head_size: 64,
      vocab_size: 100,
      max_positions: 128,
      hidden_size_per_layer_input: 0,
      enable_moe_block: false,
      num_kv_shared_layers: 0,
      attention_window_size: 16,
      final_logit_softcapping: 0,
      layer_types: [:sliding_attention, :full_attention]
    ]

    Bumblebee.configure(Bumblebee.Text.Gemma4, Keyword.merge(defaults, overrides))
  end
end
