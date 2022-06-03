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
  @spec load_config(repository(), keyword()) :: {:ok, map()} | {:error, String.t()}
  def load_config(repository, opts \\ []) do
    validate_repository!(repository)
    opts = Keyword.validate!(opts, [:module, :architecture, :revision, :cache_dir])
    module = opts[:module]
    architecture = opts[:architecture]

    with {:ok, path} <- download(repository, @config_filename, opts),
         {:ok, data} <- decode_config(path) do
      {inferred_module, inferred_architecture, inferrence_error} =
        case infer_model_type(data) do
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
      config = HuggingFace.Transformers.Config.load(config, data)
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
    "ResNetModel" => {Bumblebee.Vision.ResNet, :base},
    "ResNetForImageClassification" => {Bumblebee.Vision.ResNet, :for_image_classification}
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

  defp infer_model_type(_data) do
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
          {:ok, Axon.t(), params :: map(), config :: map()} | {:error, String.t()}
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

    with {:ok, config} <- config_response,
         module <- config.__struct__,
         model <- module.model(config),
         {:ok, params} <-
           load_params(
             model,
             repository,
             opts
             |> Keyword.take([:revision, :cache_dir])
             |> Keyword.put(:base_model_prefix, module.base_model_prefix())
           ) do
      {:ok, model, params, config}
    end
  end

  defp load_params(model, repository, opts) do
    base_model_prefix = opts[:base_model_prefix]

    # TODO: support format: :auto | :axon | :pytorch
    format = :pytorch
    filename = @params_filename[format]

    with {:ok, path} <- download(repository, filename, opts) do
      params =
        Bumblebee.Conversion.PyTorch.load_params!(model, path,
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
      input = Bumblebee.featurize(featurizer, [img])

  """
  @spec featurize(Bumblebee.Featurizer.t(), any()) :: any()
  def featurize(%module{} = featurizer, input) do
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
  @spec load_featurizer(repository(), keyword()) :: {:ok, map()} | {:error, String.t()}
  def load_featurizer(repository, opts \\ []) do
    validate_repository!(repository)
    opts = Keyword.validate!(opts, [:module, :revision, :cache_dir])
    module = opts[:module]

    with {:ok, path} <- download(repository, @featurizer_filename, opts),
         {:ok, data} <- decode_config(path) do
      module =
        module ||
          case infer_featurizer_type(data) do
            {:ok, module} -> module
            {:error, error} -> raise "#{error}, please specify the :module option"
          end

      config = struct!(module)
      config = HuggingFace.Transformers.Config.load(config, data)
      {:ok, config}
    end
  end

  @transformers_class_to_featurizer %{
    "ConvNextFeatureExtractor" => Bumblebee.Vision.ConvNextFeaturizer
  }

  defp infer_featurizer_type(%{"feature_extractor_type" => class_name}) do
    case @transformers_class_to_featurizer[class_name] do
      nil ->
        {:error,
         "could not match the class name #{inspect(class_name)} to any of the supported featurizers"}

      module ->
        {:ok, module}
    end
  end

  defp infer_featurizer_type(_data) do
    {:error, "could not infer featurizer type from the configuration"}
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
