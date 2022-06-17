defmodule Bumblebee.Text.BertTest do
  use ExUnit.Case, async: false

  describe "integration" do
    @tag :slow
    @tag :capture_log
    test "base model" do
      assert {:ok, _model, _params, config} =
               Bumblebee.load_model({:hf, "facebook/bart-base"}, architecture: :base)

      assert %Bumblebee.Text.Bart{architecture: :base} = config
    end
  end
end
