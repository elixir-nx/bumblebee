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
    test "supports sharded models" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"})

      assert {:ok, %{params: sharded_params}} =
               Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-GPT2Model-sharded"})

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(sharded_params))
    end

    test "supports .safetensors params file" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"})

      assert {:ok, %{params: safetensors_params}} =
               Bumblebee.load_model(
                 {:hf, "bumblebee-testing/tiny-random-GPT2Model-safetensors-only"}
               )

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(safetensors_params))
    end
  end
end
