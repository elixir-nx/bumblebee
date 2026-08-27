defmodule Bumblebee.Text.Gemma4Chat do
  @moduledoc """
  Builds Gemma 4 chat prompts.

  Gemma 4 checkpoints ship a `tokenizer.json` whose post-processor adds nothing, so no
  `<bos>` is prepended even with `add_special_tokens: true`. The model degenerates badly
  without it, so the prompt string must carry both `<bos>` and the turn markers. This
  module mirrors the `chat_template.jinja` shipped with the official checkpoints for the
  text-only cases.
  """

  @typedoc """
  A single chat message.

  The role is one of `:system`, `:user` or `:assistant` (`:developer` is treated as
  `:system`, and `:model` as `:assistant`); string roles are accepted too.
  """
  @type message :: %{required(:role) => atom() | String.t(), required(:content) => String.t()}

  @doc """
  Formats `messages` into a Gemma 4 prompt string.

  ## Options

    * `:add_generation_prompt` - append the `<|turn>model\\n` cue that tells the model to
      start an assistant reply. Defaults to `true`

    * `:enable_thinking` - emit the `<|think|>` control token, which turns on the model's
      reasoning mode. Defaults to `false`

  ## Examples

      iex> Bumblebee.Text.Gemma4Chat.format([%{role: :user, content: "Hi there"}])
      "<bos><|turn>user\\nHi there<turn|>\\n<|turn>model\\n"

      iex> Bumblebee.Text.Gemma4Chat.format(
      ...>   [%{role: :system, content: "You are terse."}, %{role: :user, content: "Hi"}]
      ...> )
      "<bos><|turn>system\\nYou are terse.<turn|>\\n<|turn>user\\nHi<turn|>\\n<|turn>model\\n"

  A system turn is emitted whenever there is system content or thinking is enabled, since
  the `<|think|>` token must appear at the top of the first system turn:

      iex> Bumblebee.Text.Gemma4Chat.format(
      ...>   [%{role: :user, content: "Hi"}], enable_thinking: true
      ...> )
      "<bos><|turn>system\\n<|think|>\\n<turn|>\\n<|turn>user\\nHi<turn|>\\n<|turn>model\\n"

  """
  @spec format([message()], keyword()) :: String.t()
  def format(messages, opts \\ []) do
    opts = Keyword.validate!(opts, add_generation_prompt: true, enable_thinking: false)
    thinking? = opts[:enable_thinking]

    {system, rest} = split_system(messages)

    header =
      if thinking? or system do
        "<|turn>system\n" <>
          if(thinking?, do: "<|think|>\n", else: "") <>
          (system || "") <> "<turn|>\n"
      else
        ""
      end

    turns =
      Enum.map_join(rest, fn %{role: role, content: content} ->
        "<|turn>#{role_name(role)}\n#{String.trim(content)}<turn|>\n"
      end)

    generation_prompt = if opts[:add_generation_prompt], do: "<|turn>model\n", else: ""

    "<bos>" <> header <> turns <> generation_prompt
  end

  defp split_system([%{role: role, content: content} | rest])
       when role in [:system, "system", :developer, "developer"] do
    {String.trim(content), rest}
  end

  defp split_system(messages), do: {nil, messages}

  defp role_name(role) when role in [:assistant, "assistant", :model, "model"], do: "model"
  defp role_name(role) when is_atom(role), do: Atom.to_string(role)
  defp role_name(role) when is_binary(role), do: role
end
