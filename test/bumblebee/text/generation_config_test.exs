defmodule Bumblebee.Text.GenerationConfigTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Text.GenerationConfig

  describe "config/2" do
    test "ensures either of length and new token options are set" do
      assert %GenerationConfig{max_length: 10, max_new_tokens: nil} =
               GenerationConfig.config(%GenerationConfig{max_new_tokens: 10}, max_length: 10)

      assert %GenerationConfig{max_length: nil, max_new_tokens: 10} =
               GenerationConfig.config(%GenerationConfig{max_length: 10}, max_new_tokens: 10)

      assert %GenerationConfig{min_length: 10, min_new_tokens: nil} =
               GenerationConfig.config(%GenerationConfig{min_new_tokens: 10}, min_length: 10)

      assert %GenerationConfig{min_length: nil, min_new_tokens: 10} =
               GenerationConfig.config(%GenerationConfig{min_length: 10}, min_new_tokens: 10)
    end

    test "raises if both length and new token options are set" do
      assert_raise ArgumentError,
                   "only one of :max_new_tokens or :max_length options must be given, but got both",
                   fn ->
                     GenerationConfig.config(%GenerationConfig{},
                       max_length: 10,
                       max_new_tokens: 10
                     )
                   end

      assert_raise ArgumentError,
                   "only one of :min_new_tokens or :min_length options must be given, but got both",
                   fn ->
                     GenerationConfig.config(%GenerationConfig{},
                       min_length: 10,
                       min_new_tokens: 10
                     )
                   end
    end

    test "raises on invalid strategy" do
      assert_raise ArgumentError,
                   "expected strategy type to be either :greedy_search or :contrastive_search, got: :invalid",
                   fn ->
                     GenerationConfig.config(%GenerationConfig{}, strategy: %{type: :invalid})
                   end

      assert_raise ArgumentError,
                   "expected strategy to have :type, but was not present in %{}",
                   fn ->
                     GenerationConfig.config(%GenerationConfig{}, strategy: %{})
                   end

      assert_raise ArgumentError,
                   "expected strategy to be a map, but got: :greedy_search",
                   fn ->
                     GenerationConfig.config(%GenerationConfig{}, strategy: :greedy_search)
                   end

      assert_raise ArgumentError,
                   "missing keys [:alpha, :top_k] for strategy :contrastive_search",
                   fn ->
                     GenerationConfig.config(%GenerationConfig{},
                       strategy: %{type: :contrastive_search}
                     )
                   end

      assert_raise ArgumentError,
                   "unexpected keys [:unexpected] for strategy :contrastive_search",
                   fn ->
                     GenerationConfig.config(%GenerationConfig{},
                       strategy: %{
                         type: :contrastive_search,
                         top_k: 4,
                         alpha: 0.6,
                         unexpected: true
                       }
                     )
                   end
    end
  end
end
