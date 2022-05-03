defmodule BumblebeeTest do
  use ExUnit.Case
  doctest Bumblebee

  test "greets the world" do
    assert Bumblebee.hello() == :world
  end
end
