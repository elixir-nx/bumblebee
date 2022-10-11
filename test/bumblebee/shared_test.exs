defmodule Bumblebee.SharedTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Shared

  describe "validate_label_options/1" do
    test "passes when :id_to_label is empty" do
      config = %{__struct__: TestConfig, num_labels: 3, id_to_label: %{}}

      assert Shared.validate_label_options(config) == config
    end

    test "passes when :id_to_label is matches :num_labels" do
      id_to_label = %{0 => "cat", 1 => "dog", 2 => "squirrel"}
      config = %{__struct__: TestConfig, num_labels: 3, id_to_label: id_to_label}

      assert Shared.validate_label_options(config) == config
    end

    test "raises an error if mismatched :num_labels and :id_to_label are given" do
      id_to_label = %{0 => "cat", 1 => "dog"}
      config = %{__struct__: TestConfig, num_labels: 3, id_to_label: id_to_label}

      assert_raise ArgumentError,
                   ~s/size mismatch between :num_labels (3) and :id_to_label (%{0 => "cat", 1 => "dog"})/,
                   fn ->
                     Shared.validate_label_options(config)
                   end
    end
  end
end
