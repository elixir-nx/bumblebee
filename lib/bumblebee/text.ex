defmodule Bumblebee.Text do
  @moduledoc """
  High-level tasks related to text processing.
  """

  @type token_classification_input :: String.t()
  @type token_classification_output :: %{entities: list(token_classification_entity())}

  @typedoc """
  A single entity label.

  Note that `start` and `end` indices are expressed in terms of UTF-8
  bytes.
  """
  @type token_classification_entity :: %{
          start: non_neg_integer(),
          end: non_neg_integer(),
          score: float(),
          label: String.t(),
          phrase: String.t()
        }

  @doc """
  Builds serving for token classification.

  The serving accepts `t:token_classification_input/0` and returns
  `t:token_classification_output/0`. A list of inputs is also supported.

  This function can be used for tasks such as named entity recognition
  (NER) or part of speech tagging (POS).

  The recognized entities can optionally be aggregated into groups
  based on the given strategy.

  ## Options

    * `:aggregation` - an optional strategy for aggregating adjacent
      tokens. Token classification models output probabilities for
      each possible token class. The aggregation strategy takes scores
      for each token (which possibly represents subwords) and groups
      tokens into phrases which are readily interpretable as entities
      of a certain class. Supported aggregation strategies:

        * `nil` (default) - corresponds to no aggregation and returns
          the most likely label for each input token

        * `:same` - groups adjacent tokens with the same label. If
          the labels use beginning-inside-outside (BIO) tagging, the
          boundaries are respected and the prefix is omitted in the
          output labels

    * `:ignored_labels` - the labels to ignore in the final output.
      The labels should be specified without BIO prefix. Defaults to
      `["O"]`

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  ## Examples

      {:ok, bert} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-cased"})

      serving = Bumblebee.Text.token_classification(bert, tokenizer, aggregation: :same)

      text = "Rachel Green works at Ralph Lauren in New York City in the sitcom Friends"
      Nx.Serving.run(serving, text)
      #=> %{
      #=>  entities: [
      #=>    %{end: 12, label: "PER", phrase: "Rachel Green", score: 0.9997024834156036, start: 0},
      #=>    %{end: 34, label: "ORG", phrase: "Ralph Lauren", score: 0.9968731701374054, start: 22},
      #=>    %{end: 51, label: "LOC", phrase: "New York City", score: 0.9995547334353129, start: 38},
      #=>    %{end: 73, label: "MISC", phrase: "Friends", score: 0.6997143030166626, start: 66}
      #=>  ]
      #=>}

  """
  @spec token_classification(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate token_classification(model_info, tokenizer, opts \\ []),
    to: Bumblebee.Text.TokenClassification

  @type generation_input :: String.t()
  @type generation_output :: %{results: list(generation_result())}
  @type generation_result :: %{text: String.t()}

  @doc """
  Builds serving for prompt-driven text generation.

  The serving accepts `t:generation_input/0` and returns `t:generation_output/0`.
  A list of inputs is also supported.

  Note that either `:max_new_tokens` or `:max_length` must be specified.

  ## Options

    * `:max_new_tokens` - the maximum number of tokens to be generated,
      ignoring the number of tokens in the prompt

    * `:min_new_tokens` - the minimum number of tokens to be generated,
      ignoring the number of tokens in the prompt

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  Also accepts all the other options of `Bumblebee.Text.Generation.build_generate/3`.

  ## Examples

      {:ok, gpt2} = Bumblebee.load_model({:hf, "gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})

      serving = Bumblebee.Text.generation(gpt2, tokenizer, max_new_tokens: 15)

      prompt = "Elixir is a functional"
      Nx.Serving.run(serving, prompt)
      #=> %{
      #=>   results: [
      #=>     %{
      #=>       text: "Elixir is a functional programming language that is designed to be used in a variety of applications. It"
      #=>     }
      #=>   ]
      #=> }

  """
  @spec generation(Bumblebee.model_info(), Bumblebee.Tokenizer.t(), keyword()) :: Nx.Serving.t()
  defdelegate generation(model_info, tokenizer, opts \\ []), to: Bumblebee.Text.Generation

  @type text_classification_input :: String.t()
  @type text_classification_output :: %{predictions: list(text_classification_prediction())}
  @type text_classification_prediction :: %{score: number(), label: String.t()}

  @doc """
  Builds serving for text classification.

  The serving accepts `t:text_classification_input/0` and returns
  `t:text_classification_output/0`. A list of inputs is also supported.

  ## Options

    * `:top_k` - the number of top predictions to include in the output. If
      the configured value is higher than the number of labels, all
      labels are returned. Defaults to `5`

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  ## Examples

      {:ok, bertweet} = Bumblebee.load_model({:hf, "finiteautomata/bertweet-base-sentiment-analysis"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "vinai/bertweet-base"})

      serving = Bumblebee.Text.text_classification(bertweet, tokenizer)

      text = "Cats are cute."
      Nx.Serving.run(serving, text)
      #=> %{
      #=>   predictions: [
      #=>     %{label: "POS", score: 0.9876555800437927},
      #=>     %{label: "NEU", score: 0.010068908333778381},
      #=>     %{label: "NEG", score: 0.002275536535307765}
      #=>   ]
      #=> }

  """
  @spec text_classification(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate text_classification(model_info, tokenizer, opts \\ []),
    to: Bumblebee.Text.TextClassification

  @type fill_mask_input :: String.t()
  @type fill_mask_output :: %{predictions: list(fill_mask_prediction())}
  @type fill_mask_prediction :: %{score: number(), token: String.t()}

  @doc """
  Builds serving for the fill-mask task.

  The serving accepts `t:fill_mask_input/0` and returns `t:fill_mask_output/0`.
  A list of inputs is also supported.

  In the fill-mask task, the objective is to predict a masked word in
  the text. The serving expects the input to have exactly on such word,
  denoted as `[MASK]`.

  ## Options

    * `:top_k` - the number of top predictions to include in the output.
      If the configured value is higher than the number of labels, all
      labels are returned. Defaults to `5`

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  ## Examples

      {:ok, bert} = Bumblebee.load_model({:hf, "bert-base-uncased"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

      serving = Bumblebee.Text.fill_mask(bert, tokenizer)

      text = "The capital of [MASK] is Paris."
      Nx.Serving.run(serving, text)
      #=> %{
      #=>   predictions: [
      #=>     %{score: 0.9279842972755432, token: "france"},
      #=>     %{score: 0.008412551134824753, token: "brittany"},
      #=>     %{score: 0.007433671969920397, token: "algeria"},
      #=>     %{score: 0.004957548808306456, token: "department"},
      #=>     %{score: 0.004369721747934818, token: "reunion"}
      #=>   ]
      #=> }

  """
  @spec fill_mask(Bumblebee.model_info(), Bumblebee.Tokenizer.t(), keyword()) :: Nx.Serving.t()
  defdelegate fill_mask(model_info, tokenizer, opts \\ []), to: Bumblebee.Text.FillMask

  @type zero_shot_classification_input :: String.t()
  @type zero_shot_classification_output :: %{
          predictions: list(zero_shot_classification_prediction())
        }
  @type zero_shot_classification_prediction :: %{score: number(), label: String.t()}

  @doc """
  Builds serving for the zero-shot classification task.

  The serving accepts `t:zero_shot_classification_input/0` and returns
  `t:zero_shot_classification_output/0`.

  The zero-shot task predicts zero-shot labels for a given sequence by
  proposing each label as a premise-hypothesis pairing.

  ## Options

    * `:hypothesis_template` - an arity-1 function which accepts a label
      and returns a hypothesis. The default hypothesis format is: "This example
      is #\{label\}".

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size. Note
          that the batch size refers to the number of prompts to classify,
          while the model prediction is made for every combination of
          prompt and label

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  ## Examples

      {:ok, model} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})

      labels = ["cooking", "traveling", "dancing"]
      zero_shot_serving = Bumblebee.Text.zero_shot_classification(model, tokenizer, labels)

      output = Nx.Serving.run(zero_shot_serving, "One day I will see the world")
      #=> %{
      #=>   predictions: [
      #=>     %{label: "cooking", score: 0.0070497458800673485},
      #=>     %{label: "traveling", score: 0.985000491142273},
      #=>     %{label: "dancing", score: 0.007949736900627613}
      #=>   ]
      #=> }

  """
  @spec zero_shot_classification(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          list(String.t()),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate zero_shot_classification(model_info, tokenizer, labels, opts \\ []),
    to: Bumblebee.Text.ZeroShotClassification
end
