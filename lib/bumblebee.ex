defmodule Bumblebee do
  @moduledoc """
  Pre-trained `Axon` models for easy inference and boosted training.

  Bumblebee provides ready-to-use, configurable `Axon` models. On top
  of that, it streamlines the process of loading and using pre-trained
  models by integrating with Hugging Face and
  [huggingface/transformers](https://github.com/huggingface/transformers).
  """

  alias Bumblebee.HuggingFace

  @config_filename "config.json"
  @featurizer_filename "preprocessor_config.json"
  @tokenizer_filename "tokenizer.json"
  @params_filename %{pytorch: "pytorch_model.bin"}

  @typedoc """
  A location to fetch model files from.

  Can be either:

    * `{:hf, repository_id}` - the repository on Hugging Face

    * `{:local, directory}` - the directory containing model files

  """
  @type repository :: {:hf, String.t()} | {:local, Path.t()}

  @doc """
  Builds new model configuration.

  This function is primarily useful in combination with `build_model/1`
  when building a fresh model for training.
  """
  @spec build_config(module(), atom(), keyword()) :: Bumblebee.ModelSpec.t()
  def build_config(module, architecture, config_opts \\ []) do
    config = struct!(module)
    config = %{config | architecture: architecture}
    module.config(config, config_opts)
  end

  @doc """
  Updates model configuration from options.

  This function is primarily useful for adjusting pre-trained model,
  see `load_model/2` for examples.
  """
  @spec update_config(Bumblebee.ModelSpec.t(), keyword()) :: Bumblebee.ModelSpec.t()
  def update_config(%module{} = config, config_opts \\ []) do
    module.config(config, config_opts)
  end

  @doc """
  Builds an `Axon` model according to the given configuration.

  ## Example

      config = Bumblebee.build_config(Bumblebee.Vision.ResNet, :base, embedding_size: 128)
      model = Bumblebee.build_model(config)

  """
  @spec build_model(Bumblebee.ModelSpec.t()) :: Axon.t()
  def build_model(%module{} = config) do
    module.model(config)
  end

  @doc """
  Loads model configuration from a model repository.

  ## Options

    * `:module` - the model configuration module. By default it is
      inferred from the configuration file, if that is not possible,
      it must be specified explicitly

    * `:architecture` - the model architecture, must be supported by
      `:module`. By default it is inferred from the configuration file

    * `:revision` - the specific model version to use, it can be any
      valid git identifier, such as branch name, tag name, or a commit
      hash

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

  ## Examples

      {:ok, config} = Bumblebee.load_config({:hf, "microsoft/resnet-50"})

  You can explicitly specify a different architecture:

      {:ok, config} = Bumblebee.load_config({:hf, "microsoft/resnet-50"}, architecture: :base)

  """
  @spec load_config(repository(), keyword()) ::
          {:ok, Bumblebee.ModelSpec.t()} | {:error, String.t()}
  def load_config(repository, opts \\ []) do
    validate_repository!(repository)
    opts = Keyword.validate!(opts, [:module, :architecture, :revision, :cache_dir])
    module = opts[:module]
    architecture = opts[:architecture]

    with {:ok, path} <- download(repository, @config_filename, opts),
         {:ok, model_data} <- decode_config(path) do
      {inferred_module, inferred_architecture, inferrence_error} =
        case infer_model_type(model_data) do
          {:ok, module, architecture} -> {module, architecture, nil}
          {:error, error} -> {nil, nil, error}
        end

      module = module || inferred_module
      architecture = architecture || inferred_architecture

      unless module do
        raise "#{inferrence_error}, please specify the :module and :architecture options"
      end

      architectures = module.architectures()

      if architecture && architecture not in architectures do
        raise ArgumentError,
              "expected architecture to be one of: #{Enum.map_join(architectures, ", ", &inspect/1)}, but got: #{inspect(architecture)}"
      end

      config = struct!(module)
      config = if architecture, do: %{config | architecture: architecture}, else: config
      config = HuggingFace.Transformers.Config.load(config, model_data)
      {:ok, config}
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

  @transformers_class_to_model %{
    # ResNet
    "ResNetModel" => {Bumblebee.Vision.ResNet, :base},
    "ResNetForImageClassification" => {Bumblebee.Vision.ResNet, :for_image_classification},
    # Albert
    "AlbertModel" => {Bumblebee.Text.Albert, :base},
    "AlbertForMaskedLM" => {Bumblebee.Text.Albert, :for_masked_language_modeling},
    "AlbertForSequenceClassification" => {Bumblebee.Text.Albert, :for_sequence_classification},
    "AlbertForTokenClassification" => {Bumblebee.Text.Albert, :for_token_classification},
    "AlbertForQuestionAnswering" => {Bumblebee.Text.Albert, :for_question_answering},
    "AlbertForMultipleChoice" => {Bumblebee.Text.Albert, :for_multiple_choice},
    "AlbertForPreTraining" => {Bumblebee.Text.Albert, :for_pre_training},
    # Bert
    "BertModel" => {Bumblebee.Text.Bert, :base},
    "BertForMaskedLM" => {Bumblebee.Text.Bert, :for_masked_language_modeling},
    "BertLMHeadModel" => {Bumblebee.Text.Bert, :for_causal_language_modeling},
    "BertForSequenceClassification" => {Bumblebee.Text.Bert, :for_sequence_classification},
    "BertForTokenClassification" => {Bumblebee.Text.Bert, :for_token_classification},
    "BertForQuestionAnswering" => {Bumblebee.Text.Bert, :for_question_answering},
    "BertForMultipleChoice" => {Bumblebee.Text.Bert, :for_multiple_choice},
    "BertForNextSentencePrediction" => {Bumblebee.Text.Bert, :for_next_sentence_prediction},
    "BertForPreTraining" => {Bumblebee.Text.Bert, :for_pre_training},
    # Roberta
    "RobertaModel" => {Bumblebee.Text.Roberta, :base},
    "RobertaForMaskedLM" => {Bumblebee.Text.Roberta, :for_masked_language_modeling},
    "RobertaLMHeadModel" => {Bumblebee.Text.Roberta, :for_causal_language_modeling},
    "RobertaForSequenceClassification" => {Bumblebee.Text.Roberta, :for_sequence_classification},
    "RobertaForTokenClassification" => {Bumblebee.Text.Roberta, :for_token_classification},
    "RobertaForQuestionAnswering" => {Bumblebee.Text.Roberta, :for_question_answering},
    "RobertaForMultipleChoice" => {Bumblebee.Text.Roberta, :for_multiple_choice},
    "RobertaForPreTraining" => {Bumblebee.Text.Roberta, :for_pre_training},
    # ConvNext
    "ConvNextModel" => {Bumblebee.Vision.ConvNext, :base},
    "ConvNextForImageClassification" => {Bumblebee.Vision.ConvNext, :for_image_classification},
    # ViT
    "ViTModel" => {Bumblebee.Vision.Vit, :base},
    "ViTForImageClassification" => {Bumblebee.Vision.Vit, :for_image_classification},
    "ViTForMaskedImageModeling" => {Bumblebee.Vision.Vit, :for_masked_image_modeling},
    "DeiTModel" => {Bumblebee.Vision.Deit, :base},
    "DeiTForImageClassification" => {Bumblebee.Vision.Deit, :for_image_classification},
    "DeiTForImageClassificationWithTeacher" =>
      {Bumblebee.Vision.Deit, :for_image_classification_with_teacher},
    "DeiTForMaskedImageModeling" => {Bumblebee.Vision.Deit, :for_masked_image_modeling},
    # Bart
    "BartModel" => {Bumblebee.Text.Bart, :base},
    "BartForCausalLM" => {Bumblebee.Text.Bart, :for_causal_language_modeling},
    "BartForConditionalGeneration" => {Bumblebee.Text.Bart, :for_conditional_generation},
    "BartForSequenceClassification" => {Bumblebee.Text.Bart, :for_sequence_classification},
    "BartForQuestionAnswering" => {Bumblebee.Text.Bart, :for_question_answering},
    # GPT2
    "GPT2Model" => {BumbleBee.Text.Gpt2, :base},
    "GPT2LMHeadModel" => {Bumblebee.Text.Gpt2, :for_causal_language_modeling},
    "GPT2ForTokenClassification" => {Bumblebee.Text.Gpt2, :for_token_classification}
  }

  defp infer_model_type(%{"architectures" => [class_name]}) do
    case @transformers_class_to_model[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported models"}

      {module, architecture} ->
        {:ok, module, architecture}
    end
  end

  defp infer_model_type(_model_data) do
    {:error, "could not infer model type from the configuration"}
  end

  @doc """
  Loads a pretrained model from a model repository.

  ## Options

    * `:config` - the configuration to use when building the model.
      By default the configuration is loaded from Hugging Face

    * `:module` - the model configuration module. By default it is
      inferred from the configuration file, if that is not possible,
      it must be specified explicitly

    * `:architecture` - the model architecture, must be supported by
      `:module`. By default it is inferred from the configuration file

    * `:revision` - the specific model version to use, it can be any
      valid git identifier, such as branch name, tag name, or a commit
      hash

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

  ## Examples

  By default the model type is inferred from configuration, so loading
  is as simple as:

      {:ok, model, params, config} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})

  You can explicitly specify a different architecture, in which case
  matching parameters are still loaded:

      {:ok, model, params, config} = Bumblebee.load_model({:hf, "microsoft/resnet-50"}, architecture: :base)

  To further customize the model, you can also pass the configuration:

      {:ok, config} = Bumblebee.load_config({:hf, "microsoft/resnet-50"})
      config = Bumblebee.update_config(config, num_labels: 10)
      {:ok, model, params, config} = Bumblebee.load_model({:hf, "microsoft/resnet-50"}, config: config)

  """
  @spec load_model(repository(), keyword()) ::
          {:ok, Axon.t(), params :: map(), config :: Bumblebee.ModelSpec.t()}
          | {:error, String.t()}
  def load_model(repository, opts \\ []) do
    validate_repository!(repository)

    config_response =
      if config = opts[:config] do
        {:ok, config}
      else
        load_config(
          repository,
          Keyword.take(opts, [:module, :architecture, :revision, :cache_dir])
        )
      end

    with {:ok, %module{} = config} <- config_response,
         model <- build_model(config),
         {:ok, params} <-
           load_params(
             config,
             model,
             repository,
             opts
             |> Keyword.take([:revision, :cache_dir])
             |> Keyword.put(:base_model_prefix, module.base_model_prefix())
           ) do
      {:ok, model, params, config}
    end
  end

  defp load_params(%module{} = config, model, repository, opts) do
    base_model_prefix = opts[:base_model_prefix]

    # TODO: support format: :auto | :axon | :pytorch
    format = :pytorch
    filename = @params_filename[format]

    input_template = module.input_template(config)

    with {:ok, path} <- download(repository, filename, opts) do
      params =
        Bumblebee.Conversion.PyTorch.load_params!(model, input_template, path,
          base_model_prefix: base_model_prefix
        )

      {:ok, params}
    end
  end

  @doc """
  Builds new featurizer.

  The featurizer can be then used with the `featurize/2` function to
  convert raw data into model input features.
  """
  @spec build_featurizer(module(), keyword()) :: Bumblebee.Featurizer.t()
  def build_featurizer(module, config_opts \\ []) do
    config = struct!(module)
    module.config(config, config_opts)
  end

  @doc """
  Featurizes `input` with the given featurizer.

  ## Examples

      featurizer = Bumblebee.build_featurizer(Bumblebee.Vision.ConvNextFeaturizer)
      {:ok, img} = StbImage.read_file(path)
      input = Bumblebee.apply_featurizer(featurizer, [img])

  """
  @spec apply_featurizer(Bumblebee.Featurizer.t(), any()) :: any()
  def apply_featurizer(%module{} = featurizer, input) do
    module.apply(featurizer, input)
  end

  @doc """
  Loads featurizer from a model repository.

  ## Options

    * `:module` - the featurizer module. By default it is inferred
      from the preprocessor configuration file, if that is not possible,
      it must be specified explicitly

    * `:revision` - the specific model version to use, it can be any
      valid git identifier, such as branch name, tag name, or a commit
      hash

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

  ## Examples

      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "microsoft/resnet-50"})

  """
  @spec load_featurizer(repository(), keyword()) ::
          {:ok, Bumblebee.Featurizer.t()} | {:error, String.t()}
  def load_featurizer(repository, opts \\ []) do
    validate_repository!(repository)
    opts = Keyword.validate!(opts, [:module, :revision, :cache_dir])
    module = opts[:module]

    with {:ok, path} <- download(repository, @featurizer_filename, opts),
         {:ok, featurizer_data} <- decode_config(path) do
      module =
        module ||
          case infer_featurizer_type(featurizer_data, repository, opts) do
            {:ok, module} -> module
            {:error, error} -> raise "#{error}, please specify the :module option"
          end

      config = struct!(module)
      config = HuggingFace.Transformers.Config.load(config, featurizer_data)
      {:ok, config}
    end
  end

  @transformers_class_to_featurizer %{
    "ConvNextFeatureExtractor" => Bumblebee.Vision.ConvNextFeaturizer,
    "ViTFeatureExtractor" => Bumblebee.Vision.VitFeaturizer,
    "DeiTFeatureExtractor" => Bumblebee.Vision.DeitFeaturizer
  }

  @model_type_to_featurizer %{
    "resnet" => Bumblebee.Vision.ConvNextFeaturizer,
    "convnext" => Bumblebee.Vision.ConvNextFeaturizer,
    "vit" => Bumblebee.Vision.VitFeaturizer,
    "deit" => Bumblebee.Vision.DeitFeaturizer
  }

  defp infer_featurizer_type(%{"feature_extractor_type" => class_name}, _repository, _opts) do
    case @transformers_class_to_featurizer[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported featurizers"}

      module ->
        {:ok, module}
    end
  end

  defp infer_featurizer_type(_featurizer_data, repository, opts) do
    with {:ok, path} <- download(repository, @config_filename, opts),
         {:ok, model_data} <- decode_config(path) do
      case model_data do
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

  ## Examples

      tokenizer = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})
      inputs = Bumblebee.apply_tokenizer(tokenizer, ["The capital of France is [MASK]."])

  """
  @spec apply_tokenizer(
          Bumblebee.Tokenizer.t(),
          Bumblebee.Tokenizer.input() | list(Bumblebee.Tokenizer.input()),
          keyword()
        ) :: any()
  def apply_tokenizer(%module{} = tokenizer, input, opts \\ []) do
    opts = Keyword.validate!(opts, add_special_tokens: true)
    module.apply(tokenizer, input, opts[:add_special_tokens])
  end

  @doc """
  Loads tokenizer from a model repository.

  ## Options

    * `:module` - the tokenizer module. By default it is inferred from
      the configuration files, if that is not possible, it must be
      specified explicitly

    * `:revision` - the specific model version to use, it can be any
      valid git identifier, such as branch name, tag name, or a commit
      hash

    * `:cache_dir` - the directory to store the downloaded files in.
      Defaults to the standard cache location for the given operating
      system

  ## Examples

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

  """
  @spec load_tokenizer(repository(), keyword()) ::
          {:ok, Bumblebee.Tokenizer.t()} | {:error, String.t()}
  def load_tokenizer(repository, opts \\ []) do
    validate_repository!(repository)
    opts = Keyword.validate!(opts, [:module, :revision, :cache_dir])
    module = opts[:module]

    with {:ok, path} <- download(repository, @tokenizer_filename, opts) do
      module =
        module ||
          case infer_tokenizer_type(repository, opts) do
            {:ok, module} -> module
            {:error, error} -> raise "#{error}, please specify the :module option"
          end

      config = struct!(module)
      config = HuggingFace.Transformers.Config.load(config, %{"tokenizer_file" => path})
      {:ok, config}
    end
  end

  @model_type_to_tokenizer %{
    "bert" => Bumblebee.Text.BertTokenizer,
    "roberta" => Bumblebee.Text.RobertaTokenizer,
    "albert" => Bumblebee.Text.AlbertTokenizer,
    "bart" => Bumblebee.Text.BartTokenizer,
    "gpt2" => Bumblebee.Text.Gpt2Tokenizer
  }

  defp infer_tokenizer_type(repository, opts) do
    with {:ok, path} <- download(repository, @config_filename, opts),
         {:ok, model_data} <- decode_config(path) do
      case model_data do
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

  defp download({:local, dir}, filename, _opts) do
    path = Path.join(dir, filename)

    if File.exists?(path) do
      {:ok, path}
    else
      {:error, "local file #{inspect(path)} does not exist"}
    end
  end

  defp download({:hf, repository_id}, filename, opts) do
    revision = opts[:revision]
    cache_dir = opts[:cache_dir]

    url = HuggingFace.Hub.file_url(repository_id, filename, revision)

    HuggingFace.Hub.cached_download(url, cache_dir: cache_dir)
  end

  defp validate_repository!({:hf, repository_id}) when is_binary(repository_id), do: :ok
  defp validate_repository!({:local, dir}) when is_binary(dir), do: :ok

  defp validate_repository!(other) do
    raise ArgumentError,
          "expected repository to be either {:hf, repository_id} or {:local, directory}, got: #{inspect(other)}"
  end
end
