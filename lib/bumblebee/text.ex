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

        * `:word_first` - uses `:same` strategy except that word tokens
          cannot end up with different labels. With this strategy word
          gets the label of the first token of that word when there
          is ambiguity. Note that this works only on word based models

        * `:word_average` - uses `:same` strategy except that word tokens
          cannot end up with different labels. With this strategy scores
          are averaged across word tokens and then the maximum label
          is taken. Note that this works only on word based models

        * `:word_max` - uses `:same` strategy except that word tokens
          cannot end up with different labels. With this strategy word
          gets the label of the token with the maximum score. Note that
          this works only on word based models

    * `:ignored_labels` - the labels to ignore in the final output.
      The labels should be specified without BIO prefix. Defaults to
      `["O"]`

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length.
          A list can be given, in which case the serving compiles
          a separate computation for each length and then inputs are
          matched to the smallest bounding length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:scores_function` - the function to use for converting logits to
      scores. Should be one of `:softmax`, `:sigmoid`, or `:none`.
      Defaults to `:softmax`

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

  ## Examples

      {:ok, bert} = Bumblebee.load_model({:hf, "dslim/bert-base-NER"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-cased"})

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

  @type generation_input ::
          String.t() | %{:text => String.t(), optional(:seed) => integer() | nil}
  @type generation_output :: %{results: list(generation_result())}
  @type generation_result :: %{text: String.t(), token_summary: token_summary()}
  @type token_summary :: %{
          input: pos_integer(),
          outout: pos_integer(),
          padding: non_neg_integer()
        }

  @doc """
  Builds serving for prompt-driven text generation.

  The serving accepts `t:generation_input/0` and returns `t:generation_output/0`.
  A list of inputs is also supported.

  ## Options

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length.
          A list can be given, in which case the serving compiles
          a separate computation for each length and then inputs are
          matched to the smallest bounding length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

    * `:stream` - when `true`, the serving immediately returns a
      stream that emits text chunks as they are generated. Note that
      when using streaming, only a single input can be given to the
      serving. To process a batch, call the serving with each input
      separately. Defaults to `false`

    * `:stream_done` - when `:stream` is enabled, this enables a final
      event, after all chunks have been emitted. The event has the
      shape `{:done, result}`, where `result` includes the same fields
      as `t:generation_result/0`, except for `:text`, which has been
      already streamed. Defaults to `false`

  ## Examples

      {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})
      generation_config = Bumblebee.configure(generation_config, max_new_tokens: 15)

      serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)

      Nx.Serving.run(serving, "Elixir is a functional")
      #=> %{
      #=>   results: [
      #=>     %{
      #=>       text: " programming language that is designed to be used in a variety of applications. It",
      #=>       token_summary: %{input: 5, output: 15, padding: 0}
      #=>     }
      #=>   ]
      #=> }

  We can stream the result by creating the serving with `stream: true`:

      {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})
      generation_config = Bumblebee.configure(generation_config, max_new_tokens: 15)

      serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config, stream: true)

      Nx.Serving.run(serving, "Elixir is a functional") |> Enum.to_list()
      #=> [" programming", " language", " that", " is", " designed", " to", " be", " used", " in", " a",
      #=>  " variety", " of", " applications.", " It"]

  """
  @spec generation(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          Bumblebee.Text.GenerationConfig.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate generation(model_info, tokenizer, generation_config, opts \\ []),
    to: Bumblebee.Text.TextGeneration

  @type translation_input ::
          %{
            :text => String.t(),
            :source_language_token => String.t(),
            :target_language_token => String.t(),
            optional(:seed) => integer() | nil
          }
  @type translation_output :: generation_output()

  @doc """
  Builds serving for text translation.

  The serving accepts `t:translation_input/0` and returns `t:translation_output/0`.
  A list of inputs is also supported.

  This serving is an extension of `generation/4` that handles per-input
  language configuration.

  Note that this serving is designed for multilingual models that
  require source/target language to be specified. Some text models are
  trained for specific language pairs, others expect a command such as
  "translate English to Spanish", in such cases you most likely want
  to use `generation/4`.

  ## Options

  See `generation/4` for available options.

  ## Examples

      repository_id = "facebook/nllb-200-distilled-600M"

      {:ok, model_info} = Bumblebee.load_model({:hf, repository_id})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repository_id})
      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, repository_id})

      serving = Bumblebee.Text.translation(model_info, tokenizer, generation_config)

      text = "The bank of the river is beautiful in spring"

      Nx.Serving.run(serving, %{
        text: text,
        source_language_token: "eng_Latn",
        target_language_token: "pol_Latn"
      })
      #=> %{
      #=>   results: [
      #=>     %{
      #=>       text: "W wiosnę brzeg rzeki jest piękny",
      #=>       token_summary: %{input: 11, output: 13, padding: 0}
      #=>     }
      #=>   ]
      #=> }
  """
  @spec translation(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          Bumblebee.Text.GenerationConfig.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate translation(model_info, tokenizer, generation_config, opts \\ []),
    to: Bumblebee.Text.Translation

  @type text_classification_input :: String.t() | {String.t(), String.t()}
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
          sequences are always padded/truncated to match that length.
          A list can be given, in which case the serving compiles
          a separate computation for each length and then inputs are
          matched to the smallest bounding length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:scores_function` - the function to use for converting logits to
      scores. Should be one of `:softmax`, `:sigmoid`, or `:none`.
      Defaults to `:softmax`

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

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

  @type text_embedding_input :: String.t()
  @type text_embedding_output :: %{embedding: Nx.Tensor.t()}

  @doc """
  Builds serving for text embeddings.

  The serving accepts `t:text_embedding_input/0` and returns
  `t:text_embedding_output/0`. A list of inputs is also supported.

  ## Options

    * `:output_attribute` - the attribute of the model output map to
      retrieve. When the output is a single tensor (rather than a map),
      this option is ignored. Defaults to `:pooled_state`

    * `:output_pool` - pooling to apply on top of the model output, in case
      it is not already a pooled embedding. Supported values:

        * `:mean_pooling` - performs a mean across all tokens

        * `:cls_token_pooling` - takes the embedding for the special CLS token.
          Note that we currently assume that the CLS token is the first token
          in the sequence

        * `:last_token_pooling` - takes the embedding for the last non-padding
          token in each sequence

      By default no pooling is applied

    * `:embedding_processor` - a post-processing step to apply to the
      embedding. Supported values: `:l2_norm`. By default the output is
      returned as is

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length.
          A list can be given, in which case the serving compiles
          a separate computation for each length and then inputs are
          matched to the smallest bounding length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

  ## Examples

      {:ok, model_info} = Bumblebee.load_model({:hf, "intfloat/e5-large"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "intfloat/e5-large"})

      serving = Bumblebee.Text.text_embedding(model_info, tokenizer)

      text = "query: Cats are cute."
      Nx.Serving.run(serving, text)

      #=> %{
      #=>   embedding: #Nx.Tensor<
      #=>     f32[1024]
      #=>     EXLA.Backend<host:0, 0.124908262.1234305056.185360>
      #=>     [-0.9789889454841614, -0.9814645051956177, -0.5015208125114441, 0.9867952466011047, 0.9917466640472412, -0.5557178258895874, -0.18618212640285492, 0.797040581703186, 0.8922086954116821, 0.7599573135375977, -0.16524426639080048, -0.8740050792694092, 0.9433475732803345, 0.7217797636985779, 0.9437620639801025, 0.4694959223270416, 0.40594056248664856, -0.20143413543701172, 0.7144518494606018, -0.8689796924591064, 0.94001305103302, 0.17163503170013428, -0.9896315932273865, 0.4455447494983673, 0.41139301657676697, 0.01911175064742565, -0.11275406181812286, -0.734498143196106, -0.6410953402519226, -0.628239095211029, -0.2570168673992157, 0.475137323141098, -0.7534396052360535, -0.9492156505584717, -0.17271563410758972, 0.9081271886825562, -0.4851466119289398, -0.9440935254096985, -0.20976334810256958, -0.684502899646759, -0.11581139266490936, 0.17509342730045319, 0.05547652021050453, 0.31042391061782837, 0.955132007598877, -0.35595986247062683, 0.016105204820632935, -0.3154579997062683, 0.9630348682403564, ...]
      #=>   >
      #=> }
  """
  @spec text_embedding(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate text_embedding(model_info, tokenizer, opts \\ []),
    to: Bumblebee.Text.TextEmbedding

  @type text_reranking_qwen3_input :: {String.t(), String.t()} | [{String.t(), String.t()}]
  @type text_reranking_qwen3_output :: %{
          scores: text_reranking_qwen3_score() | list(text_reranking_qwen3_score())
        }
  @type text_reranking_qwen3_score :: %{score: number(), query: String.t(), document: String.t()}

  @doc """
  Builds a serving for text reranking with Qwen3 reranker models.

  The serving expects input in one of the following formats:

    * `{query, document}` - a tuple with query and document text
    * `[{query1, doc1}, {query2, doc2}, ...]` - a list of query-document pairs

  ## Options

    * `:yes_token` - the token ID corresponding to "yes" for relevance scoring.
      If not provided, will be inferred from the tokenizer

    * `:no_token` - the token ID corresponding to "no" for relevance scoring.
      If not provided, will be inferred from the tokenizer

    * `:instruction_prefix` - the instruction prefix to use. Defaults to the
      Qwen3 reranker format

    * `:instruction_suffix` - the instruction suffix to use. Defaults to the
      Qwen3 reranker format

    * `:task_description` - the task description to include in prompts. Defaults
      to "Given a web search query, retrieve relevant passages that answer the query"

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured in `:defn_options`. You may want to set
      this option when using partitioned models on the GPU. Defaults to `false`

  ## Examples

      {:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-Reranker-0.6B"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-Reranker-0.6B"})

      serving = Bumblebee.Text.text_reranking_qwen3(model_info, tokenizer)

      query = "What is the capital of France?"
      documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany."
      ]

      pairs = Enum.map(documents, &{query, &1})
      Nx.Serving.run(serving, pairs)

  """
  @spec text_reranking_qwen3(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate text_reranking_qwen3(model_info, tokenizer, opts \\ []),
    to: Bumblebee.Text.TextRerankingQwen3

  @type fill_mask_input :: String.t()
  @type fill_mask_output :: %{predictions: list(fill_mask_prediction())}
  @type fill_mask_prediction :: %{score: number(), token: String.t()}

  @doc """
  Builds serving for the fill-mask task.

  The serving accepts `t:fill_mask_input/0` and returns `t:fill_mask_output/0`.
  A list of inputs is also supported.

  In the fill-mask task, the objective is to predict a masked word in
  the text. The serving expects the input to have exactly one such word,
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
          sequences are always padded/truncated to match that length.
          A list can be given, in which case the serving compiles
          a separate computation for each length and then inputs are
          matched to the smallest bounding length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

  ## Examples

      {:ok, bert} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})

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

  @type question_answering_input :: %{question: String.t(), context: String.t()}

  @type question_answering_output :: %{
          predictions: list(question_answering_result())
        }

  @type question_answering_result :: %{
          text: String.t(),
          start: number(),
          end: number(),
          score: number()
        }

  @doc """
  Builds serving for the question answering task.

  The serving accepts `t:question_answering_input/0` and returns
  `t:question_answering_output/0`. A list of inputs is also supported.

  The question answering task finds the most probable answer to a
  question within the given context text.

  ## Options

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size. Note
          that the batch size refers to the number of prompts to classify,
          while the model prediction is made for every combination of
          prompt and label

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length.
          A list can be given, in which case the serving compiles
          a separate computation for each length and then inputs are
          matched to the smallest bounding length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

  ## Examples

      {:ok, roberta} = Bumblebee.load_model({:hf, "deepset/roberta-base-squad2"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "FacebookAI/roberta-base"})

      serving = Bumblebee.Text.question_answering(roberta, tokenizer)

      input = %{question: "What's my name?", context: "My name is Sarah and I live in London."}
      Nx.Serving.run(serving, input)
      #=> %{results: [%{end: 16, score: 0.81039959192276, start: 11, text: "Sarah"}]}

  """
  @spec question_answering(
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate question_answering(model_info, tokenizer, opts \\ []),
    to: Bumblebee.Text.QuestionAnswering

  @type zero_shot_classification_input :: String.t()
  @type zero_shot_classification_output :: %{
          predictions: list(zero_shot_classification_prediction())
        }
  @type zero_shot_classification_prediction :: %{score: number(), label: String.t()}

  @doc """
  Builds serving for the zero-shot classification task.

  The serving accepts `t:zero_shot_classification_input/0` and returns
  `t:zero_shot_classification_output/0`. A list of inputs is also
  supported.

  The zero-shot task predicts zero-shot labels for a given sequence by
  proposing each label as a premise-hypothesis pairing.

  ## Options

    * `:top_k` - the number of top predictions to include in the output. If
      the configured value is higher than the number of labels, all
      labels are returned. Defaults to `5`

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
          sequences are always padded/truncated to match that length.
          A list can be given, in which case the serving compiles
          a separate computation for each length and then inputs are
          matched to the smallest bounding length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

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
