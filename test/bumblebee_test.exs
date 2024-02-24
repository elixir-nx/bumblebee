defmodule BumblebeeTest do
  use ExUnit.Case, async: true

  describe "load_model/2" do
    test "raises an error on invalid repository type" do
      assert_raise ArgumentError,
                   ~s/expected repository to be either {:hf, repository_id}, {:hf, repository_id, options} or {:local, directory}, got: "repo-id"/,
                   fn ->
                     Bumblebee.load_model("repo-id")
                   end
    end

    @tag :capture_log
    test "supports sharded params" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"})

      # PyTorch format

      assert {:ok, %{params: sharded_params}} =
               Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-GPT2Model-sharded"})

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(sharded_params))

      # Safetensors

      assert {:ok, %{params: sharded_params}} =
               Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-GPT2Model-sharded"},
                 params_filename: "model.safetensors"
               )

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(sharded_params))
    end

    test "supports .safetensors params" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"})

      assert {:ok, %{params: safetensors_params}} =
               Bumblebee.load_model(
                 {:hf, "bumblebee-testing/tiny-random-GPT2Model-safetensors-only"}
               )

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(safetensors_params))
    end

    test "supports params variants" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-bert-variant"},
                 params_variant: "v2"
               )

      assert {:ok, %{params: sharded_params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-bert-variant-sharded"},
                 params_variant: "v2"
               )

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(sharded_params))

      assert_raise ArgumentError,
                   ~s/parameters variant "v3" not found, available variants: "v2"/,
                   fn ->
                     Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-bert-variant"},
                       params_variant: "v3"
                     )
                   end

      assert_raise ArgumentError,
                   ~s/parameters variant "v3" not found, available variants: "v2"/,
                   fn ->
                     Bumblebee.load_model(
                       {:hf, "hf-internal-testing/tiny-random-bert-variant-sharded"},
                       params_variant: "v3"
                     )
                   end
    end

    test "passing :type casts params accordingly" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"},
                 type: :bf16
               )

      assert Nx.type(params["decoder.blocks.0.ffn.output"]["kernel"]) == {:bf, 16}
      assert Nx.type(params["decoder.blocks.0.ffn.output"]["bias"]) == {:bf, 16}

      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"},
                 type: Axon.MixedPrecision.create_policy(params: :f16)
               )

      assert Nx.type(params["decoder.blocks.0.ffn.output"]["kernel"]) == {:f, 16}
      assert Nx.type(params["decoder.blocks.0.ffn.output"]["bias"]) == {:f, 16}
    end
  end
end
