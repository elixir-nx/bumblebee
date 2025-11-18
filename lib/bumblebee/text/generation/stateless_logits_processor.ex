defmodule Bumblebee.Text.Generation.StatelessLogitsProcessor do
  @moduledoc false

  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.LogitsProcessor

  options = [
    fun: [
      default: nil,
      doc: "a state-less function that is applied to the logits"
    ]
  ]

  defstruct Bumblebee.Shared.option_defaults(options)

  @impl Bumblebee.Configurable
  def config(logits_processor, opts) do
    Bumblebee.Shared.put_config_attrs(logits_processor, opts)
  end

  @impl Bumblebee.LogitsProcessor
  def init(_logits_processor, _init_context) do
    %{}
  end

  @impl Bumblebee.LogitsProcessor
  def process(logits_processor, state, logits, process_context) do
    {state, logits_processor.fun.(logits, process_context)}
  end
end
