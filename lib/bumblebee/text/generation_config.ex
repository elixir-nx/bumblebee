defmodule Bumblebee.Text.GenerationConfig do
  alias Bumblebee.Shared

  length_options = [
    max_new_tokens: [
      default: 20,
      doc:
        "the maximum number of tokens to be generated, ignoring the number of tokens in the prompt"
    ],
    min_new_tokens: [
      default: nil,
      doc:
        "the minimum number of tokens to be generated, ignoring the number of tokens in the prompt"
    ],
    max_length: [
      default: nil,
      doc: """
      the maximum length of the sequence to be generated. Note that this length includes the
      length of the input prompt (including padding). In general, prefer `:max_new_tokens`,
      which ignores the number of tokens in the prompt
      """
    ],
    min_length: [
      default: nil,
      doc: """
      the minimum length of the sequence to be generated. Note that this length includes the
      length of the input prompt (including padding). In general, prefer `:min_new_tokens`,
      which ignores the number of tokens in the prompt
      """
    ]
  ]

  strategy_options = [
    strategy: [
      default: %{type: :greedy_search},
      doc: """
      the method deciding how tokens are selected, it has a significant impact on the quality
      of the generated sequence. Should be a map with `:type` and strategy-specific options.

        * `:greedy_search` - the most straightforward approach, where in
          every iteration the most probable token (as given by the model)
          is taken.

          Example: `%{type: :greedy_search}`.

        * `:contrastive_search` - state-of-the-art decoding method, capable
          of producing high quality, coherent sequences. The results are
          deterministic. See [this article](https://huggingface.co/blog/introducing-csearch)
          for more details.

            * `:top_k` (required) - the number of highest probability vocabulary tokens considered
              as a continuation

            * `:alpha` (required) - the weight of degeneration penalty. It balances the model
              confidence and the penalty

          Example: `%{type: :contrastive_search, top_k: 4, alpha: 0.6}`.

        * `:multinomial_sampling` - this method samples tokens according to the probability
          distribution given by the model. The results are nondeterministic, unless a seed
          is specified.

            * `:top_k` (optional) - when specified, restricts sampling to top-k most probable
              candidates

            * `:top_p` (optional) - when specified, restricts sampling to tokens which probabilities
              add up to top-p
      """
    ]
  ]

  token_options = [
    decoder_start_token_id: [
      default: nil,
      doc:
        "the id of the initial token when generating from scratch, in case of encoder-decoder models"
    ],
    forced_bos_token_id: [
      default: nil,
      doc: "the id of the token to force as the first generated token"
    ],
    forced_eos_token_id: [
      default: nil,
      doc:
        "the id of the token to force as the last generated token when `:max_length` is reached"
    ],
    forced_token_ids: [
      default: [],
      doc:
        "a list of `{index, token_id}` pairs forcing `token_id` to appear at `index` in the generated sequence"
    ],
    suppressed_token_ids: [
      default: [],
      doc: "a list of token ids to suppress during generation"
    ],
    no_repeat_ngram_length: [
      default: nil,
      doc: "when set, n-grams of the given length can occur only once in the generated sequence"
    ],
    temperature: [
      default: nil,
      doc: """
      enables exponential scaling of the output probability distribution. The temperature value effectively
      determines the randomness of the predicted tokens. Values smaller than 1.0 decrease the randomness,
      while bigger values increase it. Note that this is only relevant for generation `:strategy` that does
      sampling based on the output probability distribution
      """
    ]
  ]

  special_token_options = [
    bos_token_id: [
      default: nil,
      doc: "the id of the beginning-of-sequence token"
    ],
    eos_token_id: [
      default: nil,
      doc: "the id of the end-of-sequence token"
    ],
    pad_token_id: [
      default: nil,
      doc: "the id of the padding token"
    ]
  ]

  other_options = [
    extra_config: [
      default: nil,
      doc: "additional configuration specific to the given model"
    ]
  ]

  options =
    length_options ++ strategy_options ++ token_options ++ special_token_options ++ other_options

  @moduledoc """
  A set of configuration options controlling text generation.

  This struct is expected by `Bumblebee.Text.Generation.build_generate/3`.

  ## Configuration

  ### Options controlling length

  #{Shared.options_doc(length_options)}

  ### Options controlling strategy

  #{Shared.options_doc(strategy_options)}

  ### Options controlling generated tokens

  #{Shared.options_doc(token_options)}

  ### Special tokens used during generation

  #{Shared.options_doc(special_token_options)}
  """

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Configurable

  @type t :: %__MODULE__{}

  @impl true
  def config(config, opts \\ []) do
    opts =
      case {opts[:max_new_tokens], opts[:max_length]} do
        {nil, nil} ->
          opts

        {_, nil} ->
          put_in(opts[:max_length], nil)

        {nil, _} ->
          put_in(opts[:max_new_tokens], nil)

        _ ->
          raise ArgumentError,
                "only one of :max_new_tokens or :max_length options must be given, but got both"
      end

    opts =
      case {opts[:min_new_tokens], opts[:min_length]} do
        {nil, nil} ->
          opts

        {_, nil} ->
          put_in(opts[:min_length], nil)

        {nil, _} ->
          put_in(opts[:min_new_tokens], nil)

        _ ->
          raise ArgumentError,
                "only one of :min_new_tokens or :min_length options must be given, but got both"
      end

    with {:ok, strategy} <- Keyword.fetch(opts, :strategy) do
      validate_strategy!(strategy)
    end

    Shared.put_config_attrs(config, opts)
  end

  defp validate_strategy!(%{type: :greedy_search} = strategy) do
    validate_strategy_keys!(strategy, [:type], [])
  end

  defp validate_strategy!(%{type: :contrastive_search} = strategy) do
    validate_strategy_keys!(strategy, [:type, :top_k, :alpha], [])
  end

  defp validate_strategy!(%{type: :multinomial_sampling} = strategy) do
    validate_strategy_keys!(strategy, [:type], [:top_k, :top_p])
  end

  defp validate_strategy!(%{type: type}) do
    raise ArgumentError,
          "expected strategy type to be either :greedy_search or :contrastive_search, got: #{inspect(type)}"
  end

  defp validate_strategy!(%{} = other) do
    raise ArgumentError,
          "expected strategy to have :type, but was not present in #{inspect(other)}"
  end

  defp validate_strategy!(other) do
    raise ArgumentError, "expected strategy to be a map, but got: #{inspect(other)}"
  end

  defp validate_strategy_keys!(strategy, required_keys, optional_keys) do
    actual = strategy |> Map.keys() |> Enum.sort()

    missing_keys = Enum.sort(required_keys -- actual)

    if missing_keys != [] do
      raise ArgumentError,
            "missing keys #{inspect(missing_keys)} for strategy #{inspect(strategy.type)}"
    end

    extra_keys = Enum.sort((actual -- required_keys) -- optional_keys)

    if extra_keys != [] do
      raise ArgumentError,
            "unexpected keys #{inspect(extra_keys)} for strategy #{inspect(strategy.type)}"
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      import Shared.Converters

      # Special case joint configurations
      data =
        case data do
          %{"model_type" => "blip", "text_config" => data} -> data
          data -> data
        end

      data =
        case data do
          # During generation BLIP uses SEP token as the EOS token
          %{"model_type" => "blip_text_model", "sep_token_id" => sep_token_id} ->
            put_in(data["eos_token_id"], sep_token_id)

          data ->
            data
        end

      data =
        case data do
          %{"forced_decoder_ids" => ids} ->
            ids = Enum.reject(ids, &match?([_idx, nil], &1))
            put_in(data["forced_decoder_ids"], ids)

          data ->
            data
        end

      data =
        case data do
          %{"suppress_tokens" => nil} -> Map.delete(data, "suppress_tokens")
          data -> data
        end

      opts =
        convert!(data,
          max_new_tokens: {"max_new_tokens", optional(number())},
          min_new_tokens: {"min_new_tokens", optional(number())},
          max_length: {"max_length", optional(number())},
          min_length: {"min_length", optional(number())},
          decoder_start_token_id: {"decoder_start_token_id", optional(number())},
          bos_token_id: {"bos_token_id", optional(number())},
          eos_token_id: {"eos_token_id", optional(number())},
          pad_token_id: {"pad_token_id", optional(number())},
          forced_bos_token_id: {"forced_bos_token_id", optional(number())},
          forced_eos_token_id: {"forced_eos_token_id", optional(number())},
          forced_token_ids: {"forced_decoder_ids", list(tuple([number(), number()]))},
          suppressed_token_ids: {"suppress_tokens", list(number())},
          no_repeat_ngram_length: {"no_repeat_ngram_size", number()}
        )

      strategy_opts =
        data
        |> convert!(
          sample: {"do_sample", boolean()},
          top_k: {"top_k", optional(number())},
          top_p: {"top_p", optional(number())},
          alpha: {"penalty_alpha", optional(number())}
        )
        |> Enum.reject(fn {_key, value} -> value == nil end)
        |> Map.new()
        |> case do
          %{sample: true} = opts ->
            options =
              Map.filter(opts, fn
                {:top_k, k} when k > 0 -> true
                {:top_p, p} when p < 1.0 -> true
                _ -> false
              end)

            [strategy: Map.merge(%{type: :multinomial_sampling}, options)]

          %{top_k: top_k, alpha: alpha} when top_k > 1 and alpha > 0 ->
            [strategy: %{type: :contrastive_search, top_k: top_k, alpha: alpha}]

          _ ->
            []
        end

      @for.config(config, opts ++ strategy_opts)
    end
  end
end
