defmodule BumblebeeTest do
  use ExUnit.Case, async: true

  describe "load_model/2" do
    test "raises an error on invalid repository type" do
      assert_raise ArgumentError,
                   ~s/expected repository to be either {:hf, repository_id} or {:local, directory}, got: "repo-id"/,
                   fn ->
                     Bumblebee.load_model("repo-id")
                   end
    end
  end
end
