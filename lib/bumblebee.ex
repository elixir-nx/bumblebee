defmodule Bumblebee do
  @moduledoc """
  Pre-trained `Axon` models for easy inference and boosted training.

  Bumblebee provides state-of-the-art, configurable `Axon` models. On
  top of that, it streamlines the process of loading pre-trained models
  by integrating with Hugging Face Hub and [🤗 Transformers](https://github.com/huggingface/transformers).

  ## Usage

  You can load one of the supported models by specifying the model repository:

      {:ok, bert} = Bumblebee.load_model({:hf, "bert-base-uncased"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

  Then you are ready to make predictions:

      inputs = Bumblebee.apply_tokenizer(tokenizer, "Hello Bumblebee!")
      outputs = Axon.predict(bert.model, bert.params, inputs)

  For complete examples see the [Examples](examples.livemd) notebook.

  > #### Note {: .info}
  >
  > The models are generally large, so make sure to configure an efficient
  > `Nx` backend, such as `EXLA` or `Torchx`.
  """

  alias Bumblebee.HuggingFace

  @config_filename "config.json"
  @featurizer_filename "preprocessor_config.json"
  @tokenizer_filename "tokenizer.json"
  @tokenizer_special_tokens_filename "special_tokens_map.json"
  @scheduler_filename "scheduler_config.json"
  @params_filename %{pytorch: "pytorch_model.bin"}

  @transformers_class_to_model %{
    "AlbertForMaskedLM" => {Bumblebee.Text.Albert, :for_masked_language_modeling},
    "AlbertForMultipleChoice" => {Bumblebee.Text.Albert, :for_multiple_choice},
    "AlbertForPreTraining" => {Bumblebee.Text.Albert, :for_pre_training},
    "AlbertForQuestionAnswering" => {Bumblebee.Text.Albert, :for_question_answering},
    "AlbertForSequenceClassification" => {Bumblebee.Text.Albert, :for_sequence_classification},
    "AlbertForTokenClassification" => {Bumblebee.Text.Albert, :for_token_classification},
    "AlbertModel" => {Bumblebee.Text.Albert, :base},
    "BartForCausalLM" => {Bumblebee.Text.Bart, :for_causal_language_modeling},
    "BartForConditionalGeneration" => {Bumblebee.Text.Bart, :for_conditional_generation},
    "BartForQuestionAnswering" => {Bumblebee.Text.Bart, :for_question_answering},
    "BartForSequenceClassification" => {Bumblebee.Text.Bart, :for_sequence_classification},
    "BartModel" => {Bumblebee.Text.Bart, :base},
    "BertForMaskedLM" => {Bumblebee.Text.Bert, :for_masked_language_modeling},
    "BertForMultipleChoice" => {Bumblebee.Text.Bert, :for_multiple_choice},
    "BertForNextSentencePrediction" => {Bumblebee.Text.Bert, :for_next_sentence_prediction},
    "BertForPreTraining" => {Bumblebee.Text.Bert, :for_pre_training},
    "BertForQuestionAnswering" => {Bumblebee.Text.Bert, :for_question_answering},
    "BertForSequenceClassification" => {Bumblebee.Text.Bert, :for_sequence_classification},
    "BertForTokenClassification" => {Bumblebee.Text.Bert, :for_token_classification},
    "BertLMHeadModel" => {Bumblebee.Text.Bert, :for_causal_language_modeling},
    "BertModel" => {Bumblebee.Text.Bert, :base},
    "BlipForConditionalGeneration" => {Bumblebee.Multimodal.Blip, :for_conditional_generation},
    # These models are just RoBERTa models, but the config will list them as CamemBERT
    "CamembertModel" => {Bumblebee.Text.Roberta, :base},
    "CamembertForMaskedLM" => {Bumblebee.Text.Roberta, :for_masked_language_modeling},
    "CamembertForSequenceClassification" =>
      {Bumblebee.Text.Roberta, :for_sequence_classification},
    "CamembertForMultipleChoice" => {Bumblebee.Text.Roberta, :for_multiple_choice},
    "CamembertForTokenClassification" => {Bumblebee.Text.Roberta, :for_token_classification},
    "CamembertForQuestionAnswering" => {Bumblebee.Text.Roberta, :for_question_answering},
    "CLIPModel" => {Bumblebee.Multimodal.Clip, :base},
    "CLIPTextModel" => {Bumblebee.Text.ClipText, :base},
    "CLIPVisionModel" => {Bumblebee.Vision.ClipVision, :base},
    "ConvNextForImageClassification" => {Bumblebee.Vision.ConvNext, :for_image_classification},
    "ConvNextModel" => {Bumblebee.Vision.ConvNext, :base},
    "DeiTForImageClassification" => {Bumblebee.Vision.Deit, :for_image_classification},
    "DeiTForImageClassificationWithTeacher" =>
      {Bumblebee.Vision.Deit, :for_image_classification_with_teacher},
    "DeiTForMaskedImageModeling" => {Bumblebee.Vision.Deit, :for_masked_image_modeling},
    "DeiTModel" => {Bumblebee.Vision.Deit, :base},
    "DistilBertModel" => {Bumblebee.Text.Distilbert, :base},
    "DistilBertForMaskedLM" => {Bumblebee.Text.Distilbert, :for_masked_language_modeling},
    "DistilBertForSequenceClassification" =>
      {Bumblebee.Text.Distilbert, :for_sequence_classification},
    "DistilBertForQuestionAnswering" => {Bumblebee.Text.Distilbert, :for_question_answering},
    "DistilBertForTokenClassification" => {Bumblebee.Text.Distilbert, :for_token_classification},
    "GPT2ForSequenceClassification" => {Bumblebee.Text.Gpt2, :for_sequence_classification},
    "GPT2ForTokenClassification" => {Bumblebee.Text.Gpt2, :for_token_classification},
    "GPT2LMHeadModel" => {Bumblebee.Text.Gpt2, :for_causal_language_modeling},
    "GPT2Model" => {BumbleBee.Text.Gpt2, :base},
    "LayoutLMForMaskedLanguageModeling" =>
      {Bumblebee.Multimodal.LayoutLm, :for_masked_language_modeling},
    "LayoutLMForQuestionAnswering" => {Bumblebee.Multimodal.LayoutLm, :for_question_answering},
    "LayoutLMForSequenceClassification" =>
      {Bumblebee.Multimodal.LayoutLm, :for_sequence_classification},
    "LayoutLMForTokenClassification" =>
      {Bumblebee.Multimodal.LayoutLm, :for_token_classification},
    "LayoutLMModel" => {Bumblebee.Multimodal.LayoutLm, :base},
    "MBartForCausalLM" => {Bumblebee.Text.Mbart, :for_causal_language_modeling},
    "MBartForConditionalGeneration" => {Bumblebee.Text.Mbart, :for_conditional_generation},
    "MBartForQuestionAnswering" => {Bumblebee.Text.Mbart, :for_question_answering},
    "MBartForSequenceClassification" => {Bumblebee.Text.Mbart, :for_sequence_classification},
    "MBartModel" => {Bumblebee.Text.Mbart, :base},
    "ResNetForImageClassification" => {Bumblebee.Vision.ResNet, :for_image_classification},
    "ResNetModel" => {Bumblebee.Vision.ResNet, :base},
    "RobertaForMaskedLM" => {Bumblebee.Text.Roberta, :for_masked_language_modeling},
    "RobertaForMultipleChoice" => {Bumblebee.Text.Roberta, :for_multiple_choice},
    "RobertaForPreTraining" => {Bumblebee.Text.Roberta, :for_pre_training},
    "RobertaForQuestionAnswering" => {Bumblebee.Text.Roberta, :for_question_answering},
    "RobertaForSequenceClassification" => {Bumblebee.Text.Roberta, :for_sequence_classification},
    "RobertaForTokenClassification" => {Bumblebee.Text.Roberta, :for_token_classification},
    "RobertaForCausalLM" => {Bumblebee.Text.Roberta, :for_causal_language_modeling},
    "RobertaModel" => {Bumblebee.Text.Roberta, :base},
    "T5Model" => {Bumblebee.Text.T5, :base},
    "T5ForConditionalGeneration" => {Bumblebee.Text.T5, :for_conditional_generation},
    "ViTForImageClassification" => {Bumblebee.Vision.Vit, :for_image_classification},
    "ViTForMaskedImageModeling" => {Bumblebee.Vision.Vit, :for_masked_image_modeling},
    "ViTModel" => {Bumblebee.Vision.Vit, :base},
    # These models are just RoBERTa models, but the config will list them as XLM-RoBERTa
    "XLMRobertaForCausalLM" => {Bumblebee.Text.Roberta, :for_causal_language_modeling},
    "XLMRobertaForMaskedLM" => {Bumblebee.Text.Roberta, :for_masked_language_modeling},
    "XLMRobertaForMultipleChoice" => {Bumblebee.Text.Roberta, :for_multiple_choice},
    "XLMRobertaForQuestionAnswering" => {Bumblebee.Text.Roberta, :for_question_answering},
    "XLMRobertaForSequenceClassification" =>
      {Bumblebee.Text.Roberta, :for_sequence_classification},
    "XLMRobertaForTokenClassification" => {Bumblebee.Text.Roberta, :for_token_classification},
    "XLMRobertaModel" => {Bumblebee.Text.Roberta, :base},
    "WhisperModel" => {Bumblebee.Audio.Whisper, :base},
    "WhisperForConditionalGeneration" => {Bumblebee.Audio.Whisper, :for_conditional_generation},
    # Diffusers
    "AutoencoderKL" => {Bumblebee.Diffusion.VaeKl, :base},
    "StableDiffusionSafetyChecker" => {Bumblebee.Diffusion.StableDiffusion.SafetyChecker, :base},
    "UNet2DConditionModel" => {Bumblebee.Diffusion.UNet2DConditional, :base}
  }

  @transformers_class_to_featurizer %{
    "CLIPFeatureExtractor" => Bumblebee.Vision.ClipFeaturizer,
    "ConvNextFeatureExtractor" => Bumblebee.Vision.ConvNextFeaturizer,
    "DeiTFeatureExtractor" => Bumblebee.Vision.DeitFeaturizer,
    "ViTFeatureExtractor" => Bumblebee.Vision.VitFeaturizer,
    "WhisperFeatureExtractor" => Bumblebee.Audio.WhisperFeaturizer
  }

  @transformers_image_processor_type_to_featurizer %{
    "BlipImageProcessor" => Bumblebee.Vision.BlipFeaturizer
  }

  @model_type_to_featurizer %{
    "convnext" => Bumblebee.Vision.ConvNextFeaturizer,
    "deit" => Bumblebee.Vision.DeitFeaturizer,
    "resnet" => Bumblebee.Vision.ConvNextFeaturizer,
    "vit" => Bumblebee.Vision.VitFeaturizer,
    "whisper" => Bumblebee.Audio.WhisperFeaturizer
  }

  @model_type_to_tokenizer %{
    "albert" => Bumblebee.Text.AlbertTokenizer,
    "bart" => Bumblebee.Text.BartTokenizer,
    "bert" => Bumblebee.Text.BertTokenizer,
    "blip" => Bumblebee.Text.BertTokenizer,
    "distilbert" => Bumblebee.Text.DistilbertTokenizer,
    "camembert" => Bumblebee.Text.CamembertTokenizer,
    "clip" => Bumblebee.Text.ClipTokenizer,
    "gpt2" => Bumblebee.Text.Gpt2Tokenizer,
    "layoutlm" => Bumblebee.Text.LayoutLmTokenizer,
    "mbart" => Bumblebee.Text.MbartTokenizer,
    "roberta" => Bumblebee.Text.RobertaTokenizer,
    "t5" => Bumblebee.Text.T5Tokenizer,
    "whisper" => Bumblebee.Text.WhisperTokenizer,
    "xlm-roberta" => Bumblebee.Text.XlmRobertaTokenizer
  }

  @diffusers_class_to_scheduler %{
    "DDIMScheduler" => Bumblebee.Diffusion.DdimScheduler,
    "PNDMScheduler" => Bumblebee.Diffusion.PndmScheduler
  }

  @typedoc """
  A location to fetch model files from.

  Can be either:

    * `{:hf, repository_id}` - the repository on Hugging Face. Options
      may be passed as the third element:

        * `:revision` - the specific model version to use, it can be
          any valid git identifier, such as branch name, tag name, or
          a commit hash

        * `:cache_dir` - the directory to store the downloaded files
          in. Defaults to the standard cache location for the given
          operating system. You can also configure it globally by
          setting the `BUMBLEBEE_CACHE_DIR` environment variable

        * `:auth_token` - the token to use as HTTP bearer authorization
          for remote files

        * `:subdir` - the directory within the repository where the
          files are located

    * `{:local, directory}` - the directory containing model files

  """
  @type repository :: {:hf, String.t()} | {:hf, String.t(), keyword()} | {:local, Path.t()}

  @typedoc """
  A model together with its state and metadata.
  """
  @type model_info :: %{
          model: Axon.t(),
          params: map(),
          spec: Bumblebee.ModelSpec.t()
        }

  @doc """
  Builds or updates a configuration object with the given options.

  Expects a configuration struct or a module supporting configuration.
  These are usually configurable:

    * model specification (`Bumblebee.ModelSpec`)

    * featurizer (`Bumblebee.Featurizer`)

    * scheduler (`Bumblebee.Scheduler`)

  ## Examples

  To build a new configuration, pass a module:

      featurizer = Bumblebee.configure(Bumblebee.Vision.ConvNextFeaturizer)
      spec = Bumblebee.configure(Bumblebee.Vision.ResNet, architecture: :for_image_classification)

  Similarly, you can update an existing configuration:

      featurizer = Bumblebee.configure(featurizer, resize_method: :bilinear)
      spec = Bumblebee.configure(spec, embedding_size: 128)

  """
  @spec configure(module() | Bumblebee.Configurable.t(), keyword()) :: Bumblebee.Configurable.t()
  def configure(config, options \\ []) do
    %module{} = config = struct!(config)
    module.config(config, options)
  end

  @doc """
  Builds an `Axon` model according to the given specification.

  ## Example

      spec = Bumblebee.configure(Bumblebee.Vision.ResNet, architecture: :base, embedding_size: 128)
      model = Bumblebee.build_model(spec)

  """
  @doc type: :model
  @spec build_model(Bumblebee.ModelSpec.t()) :: Axon.t()
  def build_model(%module{} = spec) do
    module.model(spec)
  end

  @doc """
  Loads model specification from a model repository.

  ## Options

    * `:module` - the model specification module. By default it is
      inferred from the configuration file, if that is not possible,
      it must be specified explicitly

    * `:architecture` - the model architecture, must be supported by
      `:module`. By default it is inferred from the configuration file

  ## Examples

      {:ok, spec} = Bumblebee.load_spec({:hf, "microsoft/resnet-50"})

  You can explicitly specify a different architecture:

      {:ok, spec} = Bumblebee.load_spec({:hf, "microsoft/resnet-50"}, architecture: :base)

  """
  @doc type: :model
  @spec load_spec(repository(), keyword()) ::
          {:ok, Bumblebee.ModelSpec.t()} | {:error, String.t()}
  def load_spec(repository, opts \\ []) do
    repository = normalize_repository!(repository)

    opts = Keyword.validate!(opts, [:module, :architecture])

    module = opts[:module]
    architecture = opts[:architecture]

    with {:ok, path} <- download(repository, @config_filename),
         {:ok, spec_data} <- decode_config(path) do
      {inferred_module, inferred_architecture, inference_error} =
        case infer_model_type(spec_data) do
          {:ok, module, architecture} -> {module, architecture, nil}
          {:error, error} -> {nil, nil, error}
        end

      module = module || inferred_module
      architecture = architecture || inferred_architecture

      unless module do
        raise "#{inference_error}, please specify the :module and :architecture options"
      end

      architectures = module.architectures()

      if architecture && architecture not in architectures do
        raise ArgumentError,
              "expected architecture to be one of: #{Enum.map_join(architectures, ", ", &inspect/1)}, but got: #{inspect(architecture)}"
      end

      spec =
        if architecture do
          configure(module, architecture: architecture)
        else
          configure(module)
        end

      spec = HuggingFace.Transformers.Config.load(spec, spec_data)

      {:ok, spec}
    end
  end

  defp decode_config(path) do
    path
    |> File.read!()
    |> Jason.decode()
    |> case do
      {:ok, data} -> {:ok, data}
      _ -> {:error, "failed to parse the config file, it is not a valid JSON"}
    end
  end

  defp infer_model_type(%{"architectures" => [class_name]}) do
    case @transformers_class_to_model[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported models"}

      {module, architecture} ->
        {:ok, module, architecture}
    end
  end

  defp infer_model_type(%{"_class_name" => class_name}) do
    infer_model_type(%{"architectures" => [class_name]})
  end

  defp infer_model_type(_spec_data) do
    {:error, "could not infer model type from the configuration"}
  end

  @doc """
  Loads a pre-trained model from a model repository.

  ## Options

    * `:spec` - the model specification to use when building the model.
      By default the specification is loaded using `load_spec/2`

    * `:module` - the model specification module. By default it is
      inferred from the configuration file, if that is not possible,
      it must be specified explicitly

    * `:architecture` - the model architecture, must be supported by
      `:module`. By default it is inferred from the configuration file

    * `:params_filename` - the file with the model parameters to be loaded

    * `:log_params_diff` - whether to log missing, mismatched and unused
      parameters. Defaults to `true`

    * `:backend` - the backend to allocate the tensors on. It is either
      an atom or a tuple in the shape `{backend, options}`

  ## Examples

  By default the model type is inferred from configuration, so loading
  is as simple as:

      {:ok, resnet} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})
      %{model: model, params: params, spec: spec} = resnet

  You can explicitly specify a different architecture, in which case
  matching parameters are still loaded:

      {:ok, resnet} = Bumblebee.load_model({:hf, "microsoft/resnet-50"}, architecture: :base)

  To further customize the model, you can also pass the specification:

      {:ok, spec} = Bumblebee.load_spec({:hf, "microsoft/resnet-50"})
      spec = Bumblebee.configure(spec, num_labels: 10)
      {:ok, resnet} = Bumblebee.load_model({:hf, "microsoft/resnet-50"}, spec: spec)

  """
  @doc type: :model
  @spec load_model(repository(), keyword()) :: {:ok, model_info()} | {:error, String.t()}
  def load_model(repository, opts \\ []) do
    repository = normalize_repository!(repository)

    opts =
      Keyword.validate!(opts, [
        :spec,
        :module,
        :architecture,
        :params_filename,
        :backend,
        log_params_diff: true
      ])

    spec_response =
      if spec = opts[:spec] do
        {:ok, spec}
      else
        load_spec(repository, Keyword.take(opts, [:module, :architecture]))
      end

    with {:ok, spec} <- spec_response,
         model <- build_model(spec),
         {:ok, params} <-
           load_params(
             spec,
             model,
             repository,
             opts
             |> Keyword.take([:params_filename, :log_params_diff, :backend])
           ) do
      {:ok, %{model: model, params: params, spec: spec}}
    end
  end

  defp load_params(%module{} = spec, model, repository, opts) do
    # TODO: support format: :auto | :axon | :pytorch
    format = :pytorch
    filename = opts[:params_filename] || @params_filename[format]
    log_params_diff = opts[:log_params_diff]
    backend = opts[:backend]

    input_template = module.input_template(spec)

    params_mapping = Bumblebee.HuggingFace.Transformers.Model.params_mapping(spec)

    with {:ok, path} <- download(repository, filename) do
      params =
        Bumblebee.Conversion.PyTorch.load_params!(model, input_template, path,
          log_params_diff: log_params_diff,
          backend: backend,
          params_mapping: params_mapping
        )

      {:ok, params}
    end
  end

  @doc """
  Featurizes `input` with the given featurizer.

  ## Options

    * `:defn_options` - the options for JIT compilation. Note that
      this is only relevant for featurizers implemented with Nx.
      Defaults to `[]`

  ## Examples

      featurizer = Bumblebee.configure(Bumblebee.Vision.ConvNextFeaturizer)
      {:ok, img} = StbImage.read_file(path)
      inputs = Bumblebee.apply_featurizer(featurizer, [img])

  """
  @doc type: :featurizer
  @spec apply_featurizer(Bumblebee.Featurizer.t(), any(), keyword()) :: any()
  def apply_featurizer(%module{} = featurizer, input, opts \\ []) do
    opts = Keyword.validate!(opts, defn_options: [])
    module.apply(featurizer, input, opts[:defn_options])
  end

  @doc """
  Loads featurizer from a model repository.

  ## Options

    * `:module` - the featurizer module. By default it is inferred
      from the preprocessor configuration file, if that is not possible,
      it must be specified explicitly

  ## Examples

      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "microsoft/resnet-50"})

  """
  @doc type: :featurizer
  @spec load_featurizer(repository(), keyword()) ::
          {:ok, Bumblebee.Featurizer.t()} | {:error, String.t()}
  def load_featurizer(repository, opts \\ []) do
    repository = normalize_repository!(repository)
    opts = Keyword.validate!(opts, [:module])
    module = opts[:module]

    with {:ok, path} <- download(repository, @featurizer_filename),
         {:ok, featurizer_data} <- decode_config(path) do
      module =
        module ||
          case infer_featurizer_type(featurizer_data, repository) do
            {:ok, module} -> module
            {:error, error} -> raise "#{error}, please specify the :module option"
          end

      featurizer = configure(module)
      featurizer = HuggingFace.Transformers.Config.load(featurizer, featurizer_data)
      {:ok, featurizer}
    end
  end

  defp infer_featurizer_type(%{"feature_extractor_type" => class_name}, _repository) do
    case @transformers_class_to_featurizer[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported featurizers"}

      module ->
        {:ok, module}
    end
  end

  defp infer_featurizer_type(%{"image_processor_type" => class_name}, _repository) do
    case @transformers_image_processor_type_to_featurizer[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported featurizers"}

      module ->
        {:ok, module}
    end
  end

  defp infer_featurizer_type(_featurizer_data, repository) do
    with {:ok, path} <- download(repository, @config_filename),
         {:ok, featurizer_data} <- decode_config(path) do
      case featurizer_data do
        %{"model_type" => model_type} ->
          case @model_type_to_featurizer[model_type] do
            nil ->
              {:error,
               "could not match model type #{inspect(model_type)} to any of the supported featurizers"}

            module ->
              {:ok, module}
          end

        _ ->
          {:error, "could not infer featurizer type from the configuration"}
      end
    end
  end

  @doc """
  Tokenizes and encodes `input` with the given tokenizer.

  ## Options

    * `:add_special_tokens` - whether to add special tokens. Defaults
      to `true`

    * `:pad_direction` - the padding direction, either `:right` or
      `:left`. Defaults to `:right`

    * `:return_attention_mask` - whether to return attention mask for
      encoded sequence. Defaults to `true`

    * `:return_token_type_ids` - whether to return token type ids for
      encoded sequence. Defaults to `true`

    * `:return_special_tokens_mask` - whether to return special tokens
      mask for encoded sequence. Defaults to `false`

    * `:return_offsets` - whether to return token offsets for encoded
      sequence. Defaults to `false`

    * `:length` - applies fixed length padding or truncation to the given
      input if set


  ## Examples

      tokenizer = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})
      inputs = Bumblebee.apply_tokenizer(tokenizer, ["The capital of France is [MASK]."])

  """
  @doc type: :tokenizer
  @spec apply_tokenizer(
          Bumblebee.Tokenizer.t(),
          Bumblebee.Tokenizer.input() | list(Bumblebee.Tokenizer.input()),
          keyword()
        ) :: any()
  def apply_tokenizer(%module{} = tokenizer, input, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        add_special_tokens: true,
        pad_direction: :right,
        truncate_direction: :right,
        length: nil,
        return_attention_mask: true,
        return_token_type_ids: true,
        return_special_tokens_mask: false,
        return_offsets: false
      )

    module.apply(tokenizer, input, opts)
  end

  @doc """
  Loads tokenizer from a model repository.

  ## Options

    * `:module` - the tokenizer module. By default it is inferred from
      the configuration files, if that is not possible, it must be
      specified explicitly

  ## Examples

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

  """
  @doc type: :tokenizer
  @spec load_tokenizer(repository(), keyword()) ::
          {:ok, Bumblebee.Tokenizer.t()} | {:error, String.t()}
  def load_tokenizer(repository, opts \\ []) do
    repository = normalize_repository!(repository)
    opts = Keyword.validate!(opts, [:module])
    module = opts[:module]

    with {:ok, path} <- download(repository, @tokenizer_filename) do
      module =
        module ||
          case infer_tokenizer_type(repository) do
            {:ok, module} -> module
            {:error, error} -> raise "#{error}, please specify the :module option"
          end

      special_tokens_map =
        with {:ok, path} <- download(repository, @tokenizer_special_tokens_filename),
             {:ok, special_tokens_map} <- decode_config(path) do
          special_tokens_map
        else
          _ -> %{}
        end

      tokenizer = struct!(module)

      tokenizer =
        HuggingFace.Transformers.Config.load(tokenizer, %{
          "tokenizer_file" => path,
          "special_tokens_map" => special_tokens_map
        })

      {:ok, tokenizer}
    end
  end

  defp infer_tokenizer_type(repository) do
    with {:ok, path} <- download(repository, @config_filename),
         {:ok, tokenizer_data} <- decode_config(path) do
      case tokenizer_data do
        %{"model_type" => model_type} ->
          case @model_type_to_tokenizer[model_type] do
            nil ->
              {:error,
               "could not match model type #{inspect(model_type)} to any of the supported tokenizers"}

            module ->
              {:ok, module}
          end

        _ ->
          {:error, "could not infer tokenizer type from the model configuration"}
      end
    end
  end

  @doc """
  Initializes state for a new scheduler loop.

  Returns a pair of `{state, timesteps}`, where `state` is an opaque
  container expected by `scheduler_step/4` and `timesteps` is a sequence
  of subsequent timesteps for model forward pass.

  Note that the number of `timesteps` may not match `num_steps` exactly.
  `num_steps` parameterizes sampling points, however depending on the
  method, sampling certain points may require multiple forward passes
  of the model and each element in `timesteps` corresponds to a single
  forward pass.
  """
  @doc type: :scheduler
  @spec scheduler_init(
          Bumblebee.Scheduler.t(),
          non_neg_integer(),
          tuple()
        ) :: {Bumblebee.Scheduler.state(), Nx.Tensor.t()}
  def scheduler_init(%module{} = scheduler, num_steps, sample_shape) do
    module.init(scheduler, num_steps, sample_shape)
  end

  @doc """
  Predicts sample at the previous timestep using the given scheduler.

  Takes the current `sample` and `prediction` (usually noise) returned
  by the model at the current timestep. Returns `{state, prev_sample}`,
  where `state` is the updated scheduler loop state and `prev_sample`
  is the predicted sample at the previous timestep.

  Note that some schedulers require several forward passes of the model
  (and a couple calls to this function) to make an actual prediction for
  the previous sample.
  """
  @doc type: :scheduler
  @spec scheduler_step(
          Bumblebee.Scheduler.t(),
          Bumblebee.Scheduler.state(),
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) :: {Bumblebee.Scheduler.state(), Nx.Tensor.t()}
  def scheduler_step(%module{} = scheduler, state, sample, prediction) do
    module.step(scheduler, state, sample, prediction)
  end

  @doc """
  Loads scheduler from a model repository.

  ## Options

    * `:module` - the scheduler module. By default it is inferred
      from the scheduler configuration file, if that is not possible,
      it must be specified explicitly

  ## Examples

      {:ok, scheduler} =
        Bumblebee.load_scheduler({:hf, "CompVis/stable-diffusion-v1-4", subdir: "scheduler"})

  """
  @doc type: :scheduler
  @spec load_scheduler(repository(), keyword()) ::
          {:ok, Bumblebee.Scheduler.t()} | {:error, String.t()}
  def load_scheduler(repository, opts \\ []) do
    repository = normalize_repository!(repository)
    opts = Keyword.validate!(opts, [:module])
    module = opts[:module]

    with {:ok, path} <- download(repository, @scheduler_filename),
         {:ok, scheduler_data} <- decode_config(path) do
      module =
        module ||
          case infer_scheduler_type(scheduler_data) do
            {:ok, module} -> module
            {:error, error} -> raise "#{error}, please specify the :module option"
          end

      scheduler = configure(module)
      scheduler = HuggingFace.Transformers.Config.load(scheduler, scheduler_data)
      {:ok, scheduler}
    end
  end

  defp infer_scheduler_type(%{"_class_name" => class_name}) do
    case @diffusers_class_to_scheduler[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported schedulers"}

      module ->
        {:ok, module}
    end
  end

  defp infer_scheduler_type(_scheduler_data) do
    {:error, "could not infer featurizer type from the configuration"}
  end

  defp download({:local, dir}, filename) do
    path = Path.join(dir, filename)

    if File.exists?(path) do
      {:ok, path}
    else
      {:error, "local file #{inspect(path)} does not exist"}
    end
  end

  defp download({:hf, repository_id, opts}, filename) do
    revision = opts[:revision]
    cache_dir = opts[:cache_dir]
    auth_token = opts[:auth_token]
    subdir = opts[:subdir]

    filename = if subdir, do: subdir <> "/" <> filename, else: filename

    url = HuggingFace.Hub.file_url(repository_id, filename, revision)

    HuggingFace.Hub.cached_download(url, cache_dir: cache_dir, auth_token: auth_token)
  end

  defp normalize_repository!({:hf, repository_id}) when is_binary(repository_id) do
    {:hf, repository_id, []}
  end

  defp normalize_repository!({:hf, repository_id, opts}) when is_binary(repository_id) do
    opts = Keyword.validate!(opts, [:revision, :cache_dir, :auth_token, :subdir])
    {:hf, repository_id, opts}
  end

  defp normalize_repository!({:local, dir}) when is_binary(dir) do
    {:local, dir}
  end

  defp normalize_repository!(other) do
    raise ArgumentError,
          "expected repository to be either {:hf, repository_id}, {:hf, repository_id, options}" <>
            " or {:local, directory}, got: #{inspect(other)}"
  end
end
