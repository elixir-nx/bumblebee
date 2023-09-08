defmodule Bumblebee.Text.WhisperGenerationConfig do
  alias Bumblebee.Shared

  options = [
    no_timestamps_token_id: [
      default: nil,
      doc: "the id of the no-timestamps token"
    ],
    language_to_token_id: [
      default: %{},
      doc: "a map from language code to token id corresponding to that language"
    ],
    task_to_token_id: [
      default: %{},
      doc: "a map from task to token id corresponding to that task"
    ]
  ]

  @moduledoc """
  A set of Whisper-specific configuration options controlling text
  generation.

  This struct is used in the `Bumblebee.Text.GenerationConfig` struct
  under the `:extra_config` attribute.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Configurable

  @type t :: %__MODULE__{}

  @impl true
  def config(config, opts \\ []) do
    Shared.put_config_attrs(config, opts)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      import Shared.Converters

      language_converter = fn name, value ->
        with {:ok, value} <- string().(name, value) do
          {:ok,
           value
           |> String.replace_prefix("<|", "")
           |> String.replace_suffix("|>", "")}
        end
      end

      opts =
        convert!(data,
          no_timestamps_token_id: {"no_timestamps_token_id", number()},
          language_to_token_id: {"lang_to_id", map(language_converter, number())},
          task_to_token_id: {"task_to_id", map(atom(), number())}
        )

      @for.config(config, opts)
    end
  end
end
