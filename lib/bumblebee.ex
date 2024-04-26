defmodule Bumblebee do
  @external_resource "README.md"

  [_, readme_docs, _] =
    "README.md"
    |> File.read!()
    |> String.split("<!-- Docs -->")

  @moduledoc """
  Pre-trained `Axon` models for easy inference and boosted training.

  Bumblebee provides state-of-the-art, configurable `Axon` models. On
  top of that, it streamlines the process of loading pre-trained models
  by integrating with Hugging Face Hub and [🤗 Transformers](https://github.com/huggingface/transformers).

  ## Usage

  You can load one of the supported models by specifying the model repository:

      {:ok, model_info} = Bumblebee.load_model({:hf, "google-bert/bert-base-uncased"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})

  Then you are ready to make predictions:

      inputs = Bumblebee.apply_tokenizer(tokenizer, "Hello Bumblebee!")
      outputs = Axon.predict(model_info.model, model_info.params, inputs)

  ### Tasks

  On top of bare models, Bumblebee provides a number of **"servings"**
  that act as end-to-end pipelines for specific tasks.

      serving = Bumblebee.Text.fill_mask(model_info, tokenizer)
      Nx.Serving.run(serving, "The capital of [MASK] is Paris.")
      #=> %{
      #=>   predictions: [
      #=>     %{score: 0.9279842972755432, token: "france"},
      #=>     %{score: 0.008412551134824753, token: "brittany"},
      #=>     %{score: 0.007433671969920397, token: "algeria"},
      #=>     %{score: 0.004957548808306456, token: "department"},
      #=>     %{score: 0.004369721747934818, token: "reunion"}
      #=>   ]
      #=> }

  As you can see the **serving** takes care of pre-processing the
  text input, runs the model and also post-processes its output into
  more structured data. In the above example we `run` serving on the
  fly, however for production usage you can start serving as a process
  and it will automatically batch requests from multiple clients.
  Processing inputs in batches is usually much more efficient, since
  it can take advantage of parallel capabilities of the target device,
  which is particularly relevant in case of GPU. For more details read
  the `Nx.Serving` docs.

  For more examples see the [Examples](examples.livemd) notebook.

  > #### Note {: .info}
  >
  > The models are generally large, so make sure to configure an efficient
  > `Nx` backend, such as `EXLA` or `Torchx`.

  #{readme_docs}
  """

  alias Bumblebee.HuggingFace

  @config_filename "config.json"
  @featurizer_filename "preprocessor_config.json"
  @tokenizer_filename "tokenizer.json"
  @tokenizer_config_filename "tokenizer_config.json"
  @tokenizer_special_tokens_filename "special_tokens_map.json"
  @generation_filename "generation_config.json"
  @scheduler_filename "scheduler_config.json"

  @params_filenames [
    "pytorch_model.bin",
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors"
  ]

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
    "BlenderbotForConditionalGeneration" =>
      {Bumblebee.Text.Blenderbot, :for_conditional_generation},
    "BlenderbotModel" => {Bumblebee.Text.Blenderbot, :base},
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
    "ControlNetModel" => {Bumblebee.Diffusion.ControlNet, :base},
    "ConvNextForImageClassification" => {Bumblebee.Vision.ConvNext, :for_image_classification},
    "ConvNextModel" => {Bumblebee.Vision.ConvNext, :base},
    "DeiTForImageClassification" => {Bumblebee.Vision.Deit, :for_image_classification},
    "DeiTForImageClassificationWithTeacher" =>
      {Bumblebee.Vision.Deit, :for_image_classification_with_teacher},
    "DeiTForMaskedImageModeling" => {Bumblebee.Vision.Deit, :for_masked_image_modeling},
    "DeiTModel" => {Bumblebee.Vision.Deit, :base},
    "Dinov2Model" => {Bumblebee.Vision.DinoV2, :base},
    "Dinov2Backbone" => {Bumblebee.Vision.DinoV2, :backbone},
    "Dinov2ForImageClassification" => {Bumblebee.Vision.DinoV2, :for_image_classification},
    "DistilBertModel" => {Bumblebee.Text.Distilbert, :base},
    "DistilBertForMaskedLM" => {Bumblebee.Text.Distilbert, :for_masked_language_modeling},
    "DistilBertForSequenceClassification" =>
      {Bumblebee.Text.Distilbert, :for_sequence_classification},
    "DistilBertForQuestionAnswering" => {Bumblebee.Text.Distilbert, :for_question_answering},
    "DistilBertForTokenClassification" => {Bumblebee.Text.Distilbert, :for_token_classification},
    "DistilBertForMultipleChoice" => {Bumblebee.Text.Distilbert, :for_multiple_choice},
    "GemmaModel" => {Bumblebee.Text.Gemma, :base},
    "GemmaForCausalLM" => {Bumblebee.Text.Gemma, :for_causal_language_modeling},
    "GemmaForSequenceClassification" => {Bumblebee.Text.Gemma, :for_sequence_classification},
    "GPT2ForSequenceClassification" => {Bumblebee.Text.Gpt2, :for_sequence_classification},
    "GPT2ForTokenClassification" => {Bumblebee.Text.Gpt2, :for_token_classification},
    "GPT2LMHeadModel" => {Bumblebee.Text.Gpt2, :for_causal_language_modeling},
    "GPT2Model" => {Bumblebee.Text.Gpt2, :base},
    "GPTBigCodeModel" => {Bumblebee.Text.GptBigCode, :base},
    "GPTBigCodeForCausalLM" => {Bumblebee.Text.GptBigCode, :for_causal_language_modeling},
    "GPTBigCodeForSequenceClassification" =>
      {Bumblebee.Text.GptBigCode, :for_sequence_classification},
    "GPTBigCodeForTokenClassification" => {Bumblebee.Text.GptBigCode, :for_token_classification},
    "GPTNeoXModel" => {Bumblebee.Text.GptNeoX, :base},
    "GPTNeoXForCausalLM" => {Bumblebee.Text.GptNeoX, :for_causal_language_modeling},
    "GPTNeoXForSequenceClassification" => {Bumblebee.Text.GptNeoX, :for_sequence_classification},
    "GPTNeoXForTokenClassification" => {Bumblebee.Text.GptNeoX, :for_token_classification},
    "LayoutLMForMaskedLM" => {Bumblebee.Multimodal.LayoutLm, :for_masked_language_modeling},
    "LayoutLMForQuestionAnswering" => {Bumblebee.Multimodal.LayoutLm, :for_question_answering},
    "LayoutLMForSequenceClassification" =>
      {Bumblebee.Multimodal.LayoutLm, :for_sequence_classification},
    "LayoutLMForTokenClassification" =>
      {Bumblebee.Multimodal.LayoutLm, :for_token_classification},
    "LayoutLMModel" => {Bumblebee.Multimodal.LayoutLm, :base},
    "LlamaModel" => {Bumblebee.Text.Llama, :base},
    "LlamaForCausalLM" => {Bumblebee.Text.Llama, :for_causal_language_modeling},
    "LlamaForSequenceClassification" => {Bumblebee.Text.Llama, :for_sequence_classification},
    "MBartForCausalLM" => {Bumblebee.Text.Mbart, :for_causal_language_modeling},
    "MBartForConditionalGeneration" => {Bumblebee.Text.Mbart, :for_conditional_generation},
    "MBartForQuestionAnswering" => {Bumblebee.Text.Mbart, :for_question_answering},
    "MBartForSequenceClassification" => {Bumblebee.Text.Mbart, :for_sequence_classification},
    "MBartModel" => {Bumblebee.Text.Mbart, :base},
    "MistralModel" => {Bumblebee.Text.Mistral, :base},
    "MistralForCausalLM" => {Bumblebee.Text.Mistral, :for_causal_language_modeling},
    "MistralForSequenceClassification" => {Bumblebee.Text.Mistral, :for_sequence_classification},
    "PhiModel" => {Bumblebee.Text.Phi, :base},
    "PhiForCausalLM" => {Bumblebee.Text.Phi, :for_causal_language_modeling},
    "PhiForSequenceClassification" => {Bumblebee.Text.Phi, :for_sequence_classification},
    "PhiForTokenClassification" => {Bumblebee.Text.Phi, :for_token_classification},
    "Phi3ForCausalLM" => {Bumblebee.Text.Phi3, :for_causal_language_modeling},
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
    "T5EncoderModel" => {Bumblebee.Text.T5, :encoder},
    "ViTForImageClassification" => {Bumblebee.Vision.Vit, :for_image_classification},
    "ViTForMaskedImageModeling" => {Bumblebee.Vision.Vit, :for_masked_image_modeling},
    "ViTModel" => {Bumblebee.Vision.Vit, :base},
    "WhisperModel" => {Bumblebee.Audio.Whisper, :base},
    "WhisperForConditionalGeneration" => {Bumblebee.Audio.Whisper, :for_conditional_generation},
    # These models are just RoBERTa models, but the config will list them as XLM-RoBERTa
    "XLMRobertaForCausalLM" => {Bumblebee.Text.Roberta, :for_causal_language_modeling},
    "XLMRobertaForMaskedLM" => {Bumblebee.Text.Roberta, :for_masked_language_modeling},
    "XLMRobertaForMultipleChoice" => {Bumblebee.Text.Roberta, :for_multiple_choice},
    "XLMRobertaForQuestionAnswering" => {Bumblebee.Text.Roberta, :for_question_answering},
    "XLMRobertaForSequenceClassification" =>
      {Bumblebee.Text.Roberta, :for_sequence_classification},
    "XLMRobertaForTokenClassification" => {Bumblebee.Text.Roberta, :for_token_classification},
    "XLMRobertaModel" => {Bumblebee.Text.Roberta, :base},
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
    "BlipImageProcessor" => Bumblebee.Vision.BlipFeaturizer,
    "BitImageProcessor" => Bumblebee.Vision.BitFeaturizer
  }

  @model_type_to_featurizer %{
    "convnext" => Bumblebee.Vision.ConvNextFeaturizer,
    "deit" => Bumblebee.Vision.DeitFeaturizer,
    "resnet" => Bumblebee.Vision.ConvNextFeaturizer,
    "vit" => Bumblebee.Vision.VitFeaturizer,
    "whisper" => Bumblebee.Audio.WhisperFeaturizer
  }

  @model_type_to_tokenizer_type %{
    "albert" => :albert,
    "bart" => :bart,
    "bert" => :bert,
    "blenderbot" => :blenderbot,
    "blip" => :bert,
    "distilbert" => :distilbert,
    "camembert" => :camembert,
    "clip" => :clip,
    "gemma" => :gemma,
    "gpt_neox" => :gpt_neo_x,
    "gpt2" => :gpt2,
    "gpt_bigcode" => :gpt2,
    "layoutlm" => :layout_lm,
    "llama" => :llama,
    "mistral" => :llama,
    "mbart" => :mbart,
    "phi" => :code_gen,
    "phi3" => :llama,
    "roberta" => :roberta,
    "t5" => :t5,
    "whisper" => :whisper,
    "xlm-roberta" => :xlm_roberta
  }

  @diffusers_class_to_scheduler %{
    "DDIMScheduler" => Bumblebee.Diffusion.DdimScheduler,
    "LCMScheduler" => Bumblebee.Diffusion.LcmScheduler,
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

        * `:offline` - if `true`, only cached files are accessed and
          missing files result in an error. You can also configure it
          globally by setting the `BUMBLEBEE_OFFLINE` environment
          variable to `true`

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

  ## Options

    * `:type` - either a type or `Axon.MixedPrecision` policy to apply
      to the model

  ## Example

      spec = Bumblebee.configure(Bumblebee.Vision.ResNet, architecture: :base, embedding_size: 128)
      model = Bumblebee.build_model(spec)

  """
  @doc type: :model
  @spec build_model(Bumblebee.ModelSpec.t(), keyword()) :: Axon.t()
  def build_model(%module{} = spec, opts \\ []) do
    opts = Keyword.validate!(opts, [:type])

    model = module.model(spec)

    case opts[:type] do
      nil ->
        model

      %Axon.MixedPrecision.Policy{} = policy ->
        Axon.MixedPrecision.apply_policy(model, policy)

      type ->
        type = Nx.Type.normalize!(type)
        policy = Axon.MixedPrecision.create_policy(params: type, compute: type, output: type)
        Axon.MixedPrecision.apply_policy(model, policy)
    end
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

    with {:ok, repo_files} <- get_repo_files(repository) do
      do_load_spec(repository, repo_files, module, architecture)
    end
  end

  defp do_load_spec(repository, repo_files, module, architecture) do
    case repo_files do
      %{@config_filename => etag} ->
        with {:ok, path} <- download(repository, @config_filename, etag),
             {:ok, spec_data} <- decode_config(path) do
          {inferred_module, inferred_architecture, inference_error} =
            case infer_model_type(spec_data) do
              {:ok, module, architecture} -> {module, architecture, nil}
              {:error, error} -> {nil, nil, error}
            end

          module = module || inferred_module
          architecture = architecture || inferred_architecture

          unless module do
            raise ArgumentError,
                  "#{inference_error}, please specify the :module and :architecture options"
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

      %{} ->
        raise ArgumentError,
              "no config file found in the given repository. Please refer to Bumblebee" <>
                " README to learn about repositories and supported models"
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

  The model is downloaded and cached on your disk, use `cache_dir/0` to
  find the location.

  ## Parameters precision

  On GPUs computations that use numeric type of lower precision can
  be faster and use less memory, while still providing valid results.
  You can configure the model to use particular type by passing the
  `:type` option, such as `:bf16`.

  Some repositories have multiple variants of the parameter files
  with different numeric types. The variant is usually indicated in
  the file extension and you can load a particular file by specifying
  `:params_variant`, or `:params_filename`. Note however that this
  does not determine the numeric type used for inference. The file
  type is relevant in context of download bandwidth and disk space.
  If you want to use a lower precision for inference, make sure to
  also specify `:type`.

  ## Options

    * `:spec` - the model specification to use when building the model.
      By default the specification is loaded using `load_spec/2`

    * `:spec_overrides` - additional options to configure the model
      specification with. This is a shorthand for using `load_spec/2`,
      `configure/2` and passing as `:spec`

    * `:module` - the model specification module. By default it is
      inferred from the configuration file, if that is not possible,
      it must be specified explicitly

    * `:architecture` - the model architecture, must be supported by
      `:module`. By default it is inferred from the configuration file

    * `:params_variant` - when specified, instead of loading parameters
      from "<name>.<ext>", loads from "<name>.<variant>.<ext>"

    * `:params_filename` - the file with the model parameters to be loaded

    * `:log_params_diff` - whether to log missing, mismatched and unused
      parameters. By default diff is logged only if some parameters
      cannot be loaded

    * `:backend` - the backend to allocate the tensors on. It is either
      an atom or a tuple in the shape `{backend, options}`

    * `:type` - either a type or `Axon.MixedPrecision` policy to apply
      to the model. Passing this option automatically casts parameters
      to the desired type

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

  Or as a shorthand, you can pass just the options to override:

      {:ok, resnet} =
        Bumblebee.load_model({:hf, "microsoft/resnet-50"}, spec_overrides: [num_labels: 10])

  """
  @doc type: :model
  @spec load_model(repository(), keyword()) :: {:ok, model_info()} | {:error, String.t()}
  def load_model(repository, opts \\ []) do
    repository = normalize_repository!(repository)

    opts =
      Keyword.validate!(opts, [
        :spec,
        :spec_overrides,
        :module,
        :architecture,
        :params_variant,
        :params_filename,
        :log_params_diff,
        :backend,
        :type
      ])

    with {:ok, repo_files} <- get_repo_files(repository),
         {:ok, spec} <- maybe_load_model_spec(opts, repository, repo_files),
         model <- build_model(spec, Keyword.take(opts, [:type])),
         {:ok, params} <- load_params(spec, model, repository, repo_files, opts) do
      {:ok, %{model: model, params: params, spec: spec}}
    end
  end

  defp maybe_load_model_spec(opts, repository, repo_files) do
    spec_result =
      if spec = opts[:spec] do
        {:ok, spec}
      else
        do_load_spec(repository, repo_files, opts[:module], opts[:architecture])
      end

    with {:ok, spec} <- spec_result do
      if options = opts[:spec_overrides] do
        {:ok, configure(spec, options)}
      else
        {:ok, spec}
      end
    end
  end

  defp load_params(%module{} = spec, model, repository, repo_files, opts) do
    input_template = module.input_template(spec)

    params_mapping = Bumblebee.HuggingFace.Transformers.Model.params_mapping(spec)

    {filename, sharded?} =
      infer_params_filename(repo_files, opts[:params_filename], opts[:params_variant])

    loader_fun =
      filename
      |> String.replace_suffix(".index.json", "")
      |> Path.extname()
      |> params_file_loader_fun()

    with {:ok, paths} <- download_params_files(repository, repo_files, filename, sharded?) do
      opts =
        [
          params_mapping: params_mapping,
          loader_fun: loader_fun
        ] ++ Keyword.take(opts, [:backend, :log_params_diff])

      params = Bumblebee.Conversion.PyTorchParams.load_params!(model, input_template, paths, opts)
      {:ok, params}
    end
  end

  defp infer_params_filename(repo_files, nil = _filename, variant) do
    validate_variant!(repo_files, variant)

    Enum.find_value(@params_filenames, &lookup_params_filename(repo_files, &1, variant)) ||
      raise ArgumentError,
            "none of the expected parameters files found in the repository." <>
              " If the file exists under an unusual name, try specifying :params_filename"
  end

  defp infer_params_filename(repo_files, filename, variant) do
    if variant do
      IO.warn("ignoring :params_variant, because :params_filename was specified")
    end

    lookup_params_filename(repo_files, filename, nil) ||
      raise ArgumentError, "could not find file #{inspect(filename)} in the repository"
  end

  defp lookup_params_filename(repo_files, filename, variant) do
    full_filename = add_variant(filename, variant)
    full_filename_sharded = add_variant(filename <> ".index.json", variant)

    cond do
      Map.has_key?(repo_files, full_filename) -> {full_filename, false}
      Map.has_key?(repo_files, full_filename_sharded) -> {full_filename_sharded, true}
      true -> nil
    end
  end

  defp add_variant(filename, nil), do: filename

  defp add_variant(filename, variant) do
    ext = Path.extname(filename)
    base = Path.basename(filename, ext)
    base <> "." <> variant <> ext
  end

  defp validate_variant!(_repo_files, nil), do: :ok

  defp validate_variant!(repo_files, variant) do
    variants = params_variants_in_repo(repo_files)

    cond do
      variant in variants ->
        :ok

      Enum.empty?(variants) ->
        raise ArgumentError,
              "parameters variant #{inspect(variant)} not found, the repository has no variants"

      true ->
        raise ArgumentError,
              "parameters variant #{inspect(variant)} not found, available variants: " <>
                Enum.map_join(variants, ", ", &inspect/1)
    end
  end

  defp params_variants_in_repo(repo_files) do
    params_filenames = MapSet.new(@params_filenames)

    Enum.reduce(repo_files, MapSet.new(), fn {name, _etag}, variants ->
      parts = String.split(name, ".")
      {variant, parts} = List.pop_at(parts, -2)
      name = Enum.join(parts, ".")

      if String.replace_suffix(name, ".index.json", "") in params_filenames and
           not String.contains?(variant, "-of-") do
        MapSet.put(variants, variant)
      else
        variants
      end
    end)
  end

  defp download_params_files(repository, repo_files, filename, false = _sharded?) do
    with {:ok, path} <- download(repository, filename, repo_files[filename]) do
      {:ok, [path]}
    end
  end

  defp download_params_files(repository, repo_files, index_filename, true = _sharded?) do
    with {:ok, path} <- download(repository, index_filename, repo_files[index_filename]),
         {:ok, sharded_metadata} <- decode_config(path) do
      filenames =
        for {_layer, filename} <- sharded_metadata["weight_map"], uniq: true, do: filename

      Enum.reduce_while(filenames, {:ok, []}, fn filename, {:ok, paths} ->
        case download(repository, filename, repo_files[filename]) do
          {:ok, path} -> {:cont, {:ok, [path | paths]}}
          error -> {:halt, error}
        end
      end)
    end
  end

  defp params_file_loader_fun(".safetensors"), do: &Safetensors.read!(&1, lazy: true)
  defp params_file_loader_fun(_), do: &Bumblebee.Conversion.PyTorchLoader.load!/1

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

    batch = module.process_input(featurizer, input)

    if Code.ensure_loaded?(module) and function_exported?(module, :process_batch, 2) do
      Nx.Defn.jit_apply(&module.process_batch(featurizer, &1), [batch], opts[:defn_options])
    else
      batch
    end
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

    case get_repo_files(repository) do
      {:ok, %{@featurizer_filename => etag} = repo_files} ->
        with {:ok, path} <- download(repository, @featurizer_filename, etag),
             {:ok, featurizer_data} <- decode_config(path) do
          module =
            module ||
              case infer_featurizer_type(featurizer_data, repository, repo_files) do
                {:ok, module} ->
                  module

                {:error, error} ->
                  raise ArgumentError, "#{error}, please specify the :module option"
              end

          featurizer = configure(module)
          featurizer = HuggingFace.Transformers.Config.load(featurizer, featurizer_data)
          {:ok, featurizer}
        end

      {:ok, %{}} ->
        raise ArgumentError, "no featurizer found in the given repository"

      {:error, message} ->
        {:error, message}
    end
  end

  defp infer_featurizer_type(%{"feature_extractor_type" => class_name}, _repository, _repo_files) do
    case @transformers_class_to_featurizer[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported featurizers"}

      module ->
        {:ok, module}
    end
  end

  defp infer_featurizer_type(%{"image_processor_type" => class_name}, _repository, _repo_files) do
    case @transformers_image_processor_type_to_featurizer[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported featurizers"}

      module ->
        {:ok, module}
    end
  end

  defp infer_featurizer_type(_featurizer_data, repository, repo_files) do
    with {:ok, path} <- download(repository, @config_filename, repo_files[@config_filename]),
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

  ## Examples

      tokenizer = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})
      inputs = Bumblebee.apply_tokenizer(tokenizer, ["The capital of France is [MASK]."])

  """
  @doc type: :tokenizer
  @spec apply_tokenizer(
          Bumblebee.Tokenizer.t(),
          Bumblebee.Tokenizer.input() | list(Bumblebee.Tokenizer.input()),
          keyword()
        ) :: any()
  def apply_tokenizer(%module{} = tokenizer, input, opts \\ []) do
    tokenizer =
      if opts == [] do
        tokenizer
      else
        # TODO: remove options on v0.6
        IO.warn(
          "passing options to Bumblebee.apply_tokenizer/3 is deprecated," <>
            " please use Bumblebee.configure/2 to set tokenizer options"
        )

        Bumblebee.configure(tokenizer, opts)
      end

    module.apply(tokenizer, input)
  end

  @doc """
  Loads tokenizer from a model repository.

  ## Options

    * `:type` - the tokenizer type. By default it is inferred from
      the configuration files, if that is not possible, it must be
      specified explicitly

  ## Examples

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})

  """
  @doc type: :tokenizer
  @spec load_tokenizer(repository(), keyword()) ::
          {:ok, Bumblebee.Tokenizer.t()} | {:error, String.t()}
  def load_tokenizer(repository, opts \\ []) do
    repository = normalize_repository!(repository)
    opts = Keyword.validate!(opts, [:type])
    type = opts[:type]

    case get_repo_files(repository) do
      {:ok, %{@tokenizer_filename => etag} = repo_files} ->
        with {:ok, path} <- download(repository, @tokenizer_filename, etag) do
          type =
            type ||
              case infer_tokenizer_type(repository, repo_files) do
                {:ok, type} ->
                  type

                {:error, error} ->
                  raise ArgumentError, "#{error}, please specify the :module option"
              end

          tokenizer_config_result =
            if Map.has_key?(repo_files, @tokenizer_config_filename) do
              etag = repo_files[@tokenizer_config_filename]

              with {:ok, path} <- download(repository, @tokenizer_config_filename, etag) do
                decode_config(path)
              end
            else
              {:ok, %{}}
            end

          special_tokens_map_result =
            if Map.has_key?(repo_files, @tokenizer_special_tokens_filename) do
              etag = repo_files[@tokenizer_special_tokens_filename]

              with {:ok, path} <- download(repository, @tokenizer_special_tokens_filename, etag) do
                decode_config(path)
              end
            else
              {:ok, %{}}
            end

          with {:ok, tokenizer_config} <- tokenizer_config_result,
               {:ok, special_tokens_map} <- special_tokens_map_result do
            tokenizer = struct!(Bumblebee.Text.PreTrainedTokenizer, type: type)

            tokenizer =
              HuggingFace.Transformers.Config.load(tokenizer, %{
                "tokenizer_file" => path,
                # Note: special_tokens_map.json is a legacy file, now
                # tokenizer_config.json includes the same information
                # and takes precedence
                "special_tokens_map" => Map.merge(tokenizer_config, special_tokens_map)
              })

            {:ok, tokenizer}
          end
        end

      {:ok, %{@tokenizer_config_filename => _}} ->
        raise ArgumentError,
              "expected a Rust-compatible tokenizer.json file, however the repository" <>
                " includes tokenizer in a different format. Please refer to Bumblebee" <>
                " README to see the possible steps you can take"

      {:ok, %{}} ->
        raise ArgumentError, "no tokenizer found in the given repository"

      {:error, message} ->
        {:error, message}
    end
  end

  defp infer_tokenizer_type(repository, repo_files) do
    with {:ok, path} <- download(repository, @config_filename, repo_files[@config_filename]),
         {:ok, tokenizer_data} <- decode_config(path) do
      case tokenizer_data do
        %{"model_type" => model_type} ->
          case @model_type_to_tokenizer_type[model_type] do
            nil ->
              {:error,
               "could not match model type #{inspect(model_type)} to any of the supported tokenizer types"}

            type ->
              {:ok, type}
          end

        _ ->
          {:error, "could not infer tokenizer type from the model configuration"}
      end
    end
  end

  @doc """
  Loads generation config from a model repository.

  Generation config includes a number of model-specific properties,
  so it is usually best to load the config and further configure,
  rather than building from scratch.

  See `Bumblebee.Text.GenerationConfig` for all the available options.

  ## Options

    * `:spec_module` - the model specification module. By default it
      is inferred from the configuration file, if that is not possible,
      it must be specified explicitly. Some models have extra options
      related to generations and those are loaded into a separate
      struct, stored under the `:extra_config` attribute

  ## Examples

      {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

      generation_config = Bumblebee.configure(generation_config, max_new_tokens: 10)

  """
  @spec load_generation_config(repository()) ::
          {:ok, Bumblebee.Text.GenerationConfig.t()} | {:error, String.t()}
  def load_generation_config(repository, opts \\ []) do
    opts = Keyword.validate!(opts, [:spec_module])

    repository = normalize_repository!(repository)

    case get_repo_files(repository) do
      {:ok, %{@config_filename => etag} = repo_files} ->
        with {:ok, path} <- download(repository, @config_filename, etag),
             {:ok, spec_data} <- decode_config(path) do
          spec_module = opts[:spec_module]

          {inferred_module, inference_error} =
            case infer_model_type(spec_data) do
              {:ok, module, _architecture} -> {module, nil}
              {:error, error} -> {nil, error}
            end

          spec_module = spec_module || inferred_module

          unless spec_module do
            raise ArgumentError, "#{inference_error}, please specify the :spec_module option"
          end

          generation_data_result =
            if Map.has_key?(repo_files, @generation_filename) do
              etag = repo_files[@generation_filename]

              with {:ok, path} <- download(repository, @generation_filename, etag) do
                decode_config(path)
              end
            else
              # Fallback to the spec data, since it used to include
              # generation attributes
              {:ok, spec_data}
            end

          with {:ok, generation_data} <- generation_data_result do
            config = struct!(Bumblebee.Text.GenerationConfig)
            config = HuggingFace.Transformers.Config.load(config, generation_data)

            extra_config_module =
              Bumblebee.Text.Generation.extra_config_module(struct!(spec_module))

            extra_config =
              if extra_config_module do
                extra_config = struct!(extra_config_module)
                HuggingFace.Transformers.Config.load(extra_config, generation_data)
              end

            config = %{config | extra_config: extra_config}

            {:ok, config}
          end
        end

      {:error, message} ->
        {:error, message}
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
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) :: {Bumblebee.Scheduler.state(), Nx.Tensor.t()}
  def scheduler_init(%module{} = scheduler, num_steps, sample_template, prng_key) do
    module.init(scheduler, num_steps, sample_template, prng_key)
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

    case get_repo_files(repository) do
      {:ok, %{@scheduler_filename => etag}} ->
        with {:ok, path} <- download(repository, @scheduler_filename, etag),
             {:ok, scheduler_data} <- decode_config(path) do
          module =
            module ||
              case infer_scheduler_type(scheduler_data) do
                {:ok, module} ->
                  module

                {:error, error} ->
                  raise ArgumentError, "#{error}, please specify the :module option"
              end

          scheduler = configure(module)
          scheduler = HuggingFace.Transformers.Config.load(scheduler, scheduler_data)
          {:ok, scheduler}
        end

      {:ok, %{}} ->
        raise ArgumentError, "no scheduler found in the given repository"

      {:error, message} ->
        {:error, message}
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

  defp get_repo_files({:local, dir}) do
    case File.ls(dir) do
      {:ok, filenames} ->
        repo_files =
          for filename <- filenames,
              path = Path.join(dir, filename),
              File.regular?(path),
              into: %{},
              do: {filename, nil}

        {:ok, repo_files}

      {:error, reason} ->
        {:error, "could not read #{dir}, reason: #{:file.format_error(reason)}"}
    end
  end

  defp get_repo_files({:hf, repository_id, opts}) do
    subdir = opts[:subdir]
    url = HuggingFace.Hub.file_listing_url(repository_id, subdir, opts[:revision])
    cache_scope = repository_id_to_cache_scope(repository_id)

    result =
      HuggingFace.Hub.cached_download(
        url,
        [cache_scope: cache_scope] ++ Keyword.take(opts, [:cache_dir, :offline, :auth_token])
      )

    with {:ok, path} <- result,
         {:ok, data} <- decode_config(path) do
      repo_files =
        for entry <- data, entry["type"] == "file", into: %{} do
          path = entry["path"]

          name =
            if subdir do
              String.replace_leading(path, subdir <> "/", "")
            else
              path
            end

          etag_content = entry["lfs"]["oid"] || entry["oid"]
          etag = <<?", etag_content::binary, ?">>
          {name, etag}
        end

      {:ok, repo_files}
    end
  end

  defp download({:local, dir}, filename, _etag) do
    path = Path.join(dir, filename)

    if File.exists?(path) do
      {:ok, path}
    else
      {:error, "local file #{inspect(path)} does not exist"}
    end
  end

  defp download({:hf, repository_id, opts}, filename, etag) do
    filename =
      if subdir = opts[:subdir] do
        subdir <> "/" <> filename
      else
        filename
      end

    url = HuggingFace.Hub.file_url(repository_id, filename, opts[:revision])
    cache_scope = repository_id_to_cache_scope(repository_id)

    HuggingFace.Hub.cached_download(
      url,
      [etag: etag, cache_scope: cache_scope] ++
        Keyword.take(opts, [:cache_dir, :offline, :auth_token])
    )
  end

  defp repository_id_to_cache_scope(repository_id) do
    repository_id
    |> String.replace("/", "--")
    |> String.replace(~r/[^\w-]/, "")
  end

  defp normalize_repository!({:hf, repository_id}) when is_binary(repository_id) do
    {:hf, repository_id, []}
  end

  defp normalize_repository!({:hf, repository_id, opts}) when is_binary(repository_id) do
    opts = Keyword.validate!(opts, [:revision, :cache_dir, :offline, :auth_token, :subdir])
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

  @doc """
  Returns the directory where downloaded files are stored.

  Defaults to the standard cache location for the given operating system.
  Can be configured with the `BUMBLEBEE_CACHE_DIR` environment variable.
  """
  @spec cache_dir() :: String.t()
  def cache_dir() do
    if dir = System.get_env("BUMBLEBEE_CACHE_DIR") do
      Path.expand(dir)
    else
      :filename.basedir(:user_cache, "bumblebee")
    end
  end
end
