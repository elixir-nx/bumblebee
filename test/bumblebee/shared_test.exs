defmodule Bumblebee.SharedTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Shared

  describe "rotary_embedding_options_from_transformers/1" do
    test "accepts the legacy type key" do
      assert [rotary_embedding_scaling_strategy: %{type: :linear, factor: 2.0}] =
               Shared.rotary_embedding_options_from_transformers(%{
                 "rope_scaling" => %{"type" => "linear", "factor" => 2.0}
               })
    end

    test "accepts the rope_type key for YaRN" do
      assert [
               rotary_embedding_scaling_strategy: %{
                 type: :yarn,
                 factor: 4.0,
                 original_max_positions: 128,
                 beta_fast: 32.0,
                 beta_slow: 1.0
               }
             ] =
               Shared.rotary_embedding_options_from_transformers(%{
                 "max_position_embeddings" => 512,
                 "rope_parameters" => %{
                   "rope_type" => "yarn",
                   "factor" => 4.0,
                   "original_max_position_embeddings" => 128,
                   "beta_fast" => 32.0,
                   "beta_slow" => 1.0
                 }
               })
    end

    test "raises for an unsupported RoPE type" do
      assert_raise RuntimeError, ~r/unsupported rotary embedding parameters/, fn ->
        Shared.rotary_embedding_options_from_transformers(%{
          "rope_scaling" => %{"type" => "unsupported", "factor" => 2.0}
        })
      end
    end

    test "raises for conflicting type aliases" do
      assert_raise RuntimeError, ~r/conflicting "rope_type" and "type" values/, fn ->
        Shared.rotary_embedding_options_from_transformers(%{
          "rope_scaling" => %{"rope_type" => "linear", "type" => "dynamic", "factor" => 2.0}
        })
      end
    end

    test "raises for invalid YaRN parameters" do
      assert_raise RuntimeError, ~r/requires a numeric "factor" >= 1/, fn ->
        Shared.rotary_embedding_options_from_transformers(%{
          "rope_scaling" => %{
            "type" => "yarn",
            "factor" => 0.5,
            "original_max_position_embeddings" => 128
          }
        })
      end
    end

    test "rejects scaling parameters without a type" do
      assert_raise RuntimeError, ~r/scaling parameters require "type" or "rope_type"/, fn ->
        Shared.rotary_embedding_options_from_transformers(%{
          "rope_scaling" => %{"factor" => 2.0}
        })
      end
    end

    test "rejects a non-positive rope theta" do
      assert_raise RuntimeError, ~r/"rope_theta" must be a positive number/, fn ->
        Shared.rotary_embedding_options_from_transformers(%{"rope_theta" => 0})
      end
    end

    test "preserves legacy Phi-3 YaRN LongRoPE without a factor" do
      assert [
               rotary_embedding_scaling_strategy: %{
                 type: :longrope,
                 short_factor: [1.0, 1.0],
                 long_factor: [2.0, 2.0],
                 original_max_positions: 128
               }
             ] =
               Shared.rotary_embedding_options_from_transformers(%{
                 "max_position_embeddings" => 512,
                 "rope_scaling" => %{
                   "type" => "yarn",
                   "short_factor" => [1.0, 1.0],
                   "long_factor" => [2.0, 2.0],
                   "original_max_position_embeddings" => 128
                 }
               })
    end
  end

  describe "bidirectional_attention_options_from_transformers/1" do
    test "uses is_causal when present" do
      assert [use_bidirectional_attention: true] =
               Shared.bidirectional_attention_options_from_transformers(%{
                 "is_causal" => false,
                 "use_bidirectional_attention" => false
               })
    end

    test "falls back to use_bidirectional_attention" do
      assert [use_bidirectional_attention: true] =
               Shared.bidirectional_attention_options_from_transformers(%{
                 "use_bidirectional_attention" => true
               })
    end

    test "rejects a non-boolean is_causal value" do
      assert_raise RuntimeError, ~r/expected "is_causal" to be a boolean/, fn ->
        Shared.bidirectional_attention_options_from_transformers(%{"is_causal" => "false"})
      end
    end
  end

  describe "validate_label_options/1" do
    test "passes when :id_to_label is empty" do
      spec = %{__struct__: TestConfig, num_labels: 3, id_to_label: %{}}

      assert Shared.validate_label_options(spec) == spec
    end

    test "passes when :id_to_label is matches :num_labels" do
      id_to_label = %{0 => "cat", 1 => "dog", 2 => "squirrel"}
      spec = %{__struct__: TestConfig, num_labels: 3, id_to_label: id_to_label}

      assert Shared.validate_label_options(spec) == spec
    end

    test "raises an error if mismatched :num_labels and :id_to_label are given" do
      id_to_label = %{0 => "cat", 1 => "dog"}
      spec = %{__struct__: TestConfig, num_labels: 3, id_to_label: id_to_label}

      assert_raise ArgumentError,
                   ~s/size mismatch between :num_labels (3) and :id_to_label (%{0 => "cat", 1 => "dog"})/,
                   fn ->
                     Shared.validate_label_options(spec)
                   end
    end
  end
end
