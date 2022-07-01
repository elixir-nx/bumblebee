defmodule Bumblebee.SharedTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Shared

  describe "add_common_computed_options/1" do
    test "computes :num_labels and :label2id when :id2label is present" do
      id2label = %{0 => "cat", 1 => "dog", 2 => "squirrel"}
      opts = [id2label: id2label]

      opts = Shared.add_common_computed_options(opts)
      assert opts[:num_labels] == 3
      assert opts[:id2label] == id2label
      assert opts[:label2id] == %{"cat" => 0, "dog" => 1, "squirrel" => 2}
    end

    test "computes :id2label and :label2id when :num_labels is present" do
      opts = [num_labels: 3]

      opts = Shared.add_common_computed_options(opts)
      assert opts[:num_labels] == 3
      assert opts[:id2label] == %{}
      assert opts[:label2id] == %{}
    end

    test "raises an error if mismatched :num_labels and :id2label are given" do
      id2label = %{0 => "cat", 1 => "dog"}
      opts = [num_labels: 3, id2label: id2label]

      assert_raise ArgumentError,
                   ~s/size mismatch between :num_labels (3) and :id2label (%{0 => "cat", 1 => "dog"})/,
                   fn ->
                     Shared.add_common_computed_options(opts)
                   end
    end
  end

  describe "convert_to_atom/2" do
    test "converts specified keys to atoms" do
      assert Shared.convert_to_atom(
               %{"key1" => "value", "key2" => 1, "key3" => "value"},
               ["key3"]
             ) == %{"key1" => "value", "key2" => 1, "key3" => :value}
    end

    test "leaves nils unchanged" do
      assert Shared.convert_to_atom(%{"key" => nil}, ["key"]) == %{"key" => nil}
    end
  end

  describe "convert_common/1" do
    test "converts id2label keys to numbers" do
      assert Shared.convert_common(%{
               "id2label" => %{"0" => "cat", "1" => "dog"}
             }) == %{"id2label" => %{0 => "cat", 1 => "dog"}}
    end
  end
end
