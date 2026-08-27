defmodule Bumblebee.Text.Gemma4ChatTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Text.Gemma4Chat

  doctest Gemma4Chat

  # Expected values below are the output of the official chat_template.jinja shipped with
  # google/gemma-4-E2B-it, rendered via apply_chat_template(add_generation_prompt=True).

  test "user turn" do
    assert Gemma4Chat.format([%{role: :user, content: "Hi there"}]) ==
             "<bos><|turn>user\nHi there<turn|>\n<|turn>model\n"
  end

  test "thinking is emitted in a synthetic system turn when there is no system message" do
    assert Gemma4Chat.format([%{role: :user, content: "Hi there"}], enable_thinking: true) ==
             "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi there<turn|>\n<|turn>model\n"
  end

  test "system turn" do
    messages = [
      %{role: :system, content: "You are terse."},
      %{role: :user, content: "Hi there"}
    ]

    assert Gemma4Chat.format(messages) ==
             "<bos><|turn>system\nYou are terse.<turn|>\n<|turn>user\nHi there<turn|>\n<|turn>model\n"
  end

  test "thinking token precedes the system content" do
    messages = [
      %{role: :system, content: "You are terse."},
      %{role: :user, content: "Hi there"}
    ]

    assert Gemma4Chat.format(messages, enable_thinking: true) ==
             "<bos><|turn>system\n<|think|>\nYou are terse.<turn|>\n<|turn>user\nHi there<turn|>\n<|turn>model\n"
  end

  test "assistant turns are rendered with the model role" do
    messages = [
      %{role: :user, content: "A?"},
      %{role: :assistant, content: "B."},
      %{role: :user, content: "C?"}
    ]

    assert Gemma4Chat.format(messages) ==
             "<bos><|turn>user\nA?<turn|>\n<|turn>model\nB.<turn|>\n<|turn>user\nC?<turn|>\n<|turn>model\n"
  end

  test "system, multi-turn and thinking combined" do
    messages = [
      %{role: :system, content: "Sys."},
      %{role: :user, content: "A?"},
      %{role: :assistant, content: "B."},
      %{role: :user, content: "C?"}
    ]

    assert Gemma4Chat.format(messages, enable_thinking: true) ==
             "<bos><|turn>system\n<|think|>\nSys.<turn|>\n<|turn>user\nA?<turn|>\n" <>
               "<|turn>model\nB.<turn|>\n<|turn>user\nC?<turn|>\n<|turn>model\n"
  end

  test "add_generation_prompt: false omits the trailing model cue" do
    assert Gemma4Chat.format([%{role: :user, content: "Hi"}], add_generation_prompt: false) ==
             "<bos><|turn>user\nHi<turn|>\n"
  end

  test "string roles and the developer alias are accepted" do
    messages = [
      %{role: "developer", content: "Sys."},
      %{role: "user", content: "A?"},
      %{role: "assistant", content: "B."}
    ]

    assert Gemma4Chat.format(messages) ==
             "<bos><|turn>system\nSys.<turn|>\n<|turn>user\nA?<turn|>\n<|turn>model\nB.<turn|>\n" <>
               "<|turn>model\n"
  end

  test "content is trimmed, matching the template's trim filter" do
    assert Gemma4Chat.format([%{role: :user, content: "  Hi there \n"}]) ==
             "<bos><|turn>user\nHi there<turn|>\n<|turn>model\n"
  end

  test "rejects unknown options" do
    assert_raise ArgumentError, fn ->
      Gemma4Chat.format([%{role: :user, content: "Hi"}], enable_thinkng: true)
    end
  end
end
