defmodule Bumblebee.SharedTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Shared

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

  describe "rotary_embedding_options_from_transformers/1" do
    test "loads YaRN configuration" do
      data = %{
        "rope_theta" => 10_000.0,
        "rope_scaling" => %{
          "rope_type" => "yarn",
          "factor" => 4.0,
          "original_max_position_embeddings" => 2048
        }
      }

      assert Shared.rotary_embedding_options_from_transformers(data) == [
               rotary_embedding_base: 10_000.0,
               rotary_embedding_scaling_strategy: %{
                 type: :yarn,
                 factor: 4.0,
                 original_max_positions: 2048,
                 beta_fast: 32.0,
                 beta_slow: 1.0,
                 attention_factor: 0.1 * :math.log(4.0) + 1.0,
                 truncate: true
               }
             ]
    end
  end
end
