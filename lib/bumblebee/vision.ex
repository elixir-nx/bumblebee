defmodule Bumblebee.Vision do
  @moduledoc """
  High-level tasks related to vision.
  """

  @typedoc """
  A term representing an image.

  Either `Nx.Tensor` or a struct implementing `Nx.Container` and
  resolving to a tensor, with the following properties:

    * HWC order
    * RGB color channels
    * alpha channel may be present, but it's usually stripped out
    * integer type (`:s` or `:u`)

  """
  @type image :: Nx.Container.t()

  @type image_classification_input :: image()
  @type image_classification_output :: %{predictions: list(image_classification_prediction())}
  @type image_classification_prediction :: %{score: number(), label: String.t()}

  @doc """
  Builds serving for image classification.

  The serving accepts `t:image_classification_input/0` and returns
  `t:image_classification_output/0`. A list of inputs is also supported.

  ## Options

    * `:top_k` - the number of top predictions to include in the output. If
      the configured value is higher than the number of labels, all
      labels are returned. Defaults to `5`

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

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

      {:ok, resnet} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "microsoft/resnet-50"})

      serving = Bumblebee.Vision.image_classification(resnet, featurizer)

      image = StbImage.read_file!(path)
      Nx.Serving.run(serving, image)
      #=> %{
      #=>   predictions: [
      #=>     %{label: "Egyptian cat", score: 0.979233980178833},
      #=>     %{label: "tabby, tabby cat", score: 0.00679466687142849},
      #=>     %{label: "tiger cat", score: 0.005290505941957235},
      #=>     %{label: "lynx, catamount", score: 0.004550771787762642},
      #=>     %{label: "Siamese cat, Siamese", score: 1.1611092486418784e-4}
      #=>   ]
      #=> }

  """
  @spec image_classification(
          Bumblebee.model_info(),
          Bumblebee.Featurizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate image_classification(model_info, featurizer, opts \\ []),
    to: Bumblebee.Vision.ImageClassification

  @type image_to_text_input :: image() | %{:image => image(), optional(:seed) => integer() | nil}
  @type image_to_text_output :: %{results: list(image_to_text_result())}
  @type image_to_text_result :: %{text: String.t()}

  @doc """
  Builds serving for image-to-text generation.

  The serving accepts `t:image_to_text_input/0` and returns
  `t:image_to_text_output/0`. A list of inputs is also supported.

  ## Options

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

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

      {:ok, blip} = Bumblebee.load_model({:hf, "Salesforce/blip-image-captioning-base"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "Salesforce/blip-image-captioning-base"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Salesforce/blip-image-captioning-base"})

      {:ok, generation_config} =
        Bumblebee.load_generation_config({:hf, "Salesforce/blip-image-captioning-base"})

      serving =
        Bumblebee.Vision.image_to_text(blip, featurizer, tokenizer, generation_config,
          defn_options: [compiler: EXLA]
        )

      image = StbImage.read_file!(path)
      Nx.Serving.run(serving, image)
      #=> %{results: [%{text: "a cat sitting on a chair"}]}

  """
  @spec image_to_text(
          Bumblebee.model_info(),
          Bumblebee.Featurizer.t(),
          Bumblebee.Tokenizer.t(),
          Bumblebee.Text.GenerationConfig.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate image_to_text(model_info, featurizer, tokenizer, generation_config, opts \\ []),
    to: Bumblebee.Vision.ImageToText

  @type image_embedding_input :: image()
  @type image_embedding_output :: %{embedding: Nx.Tensor.t()}
  @doc """
  Builds serving for image embeddings.

  The serving accepts `t:image_embedding_input/0` and returns
  `t:image_embedding_output/0`. A list of inputs is also supported.

  ## Options

    * `:output_attribute` - the attribute of the model output map to
      retrieve. When the output is a single tensor (rather than a map),
      this option is ignored. Defaults to `:pooled_state`

    * `:embedding_processor` - a post-processing step to apply to the
      embedding. Supported values: `:l2_norm`. By default the output is
      returned as is

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

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

      {:ok, clip} =
        Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"},
          module: Bumblebee.Vision.ClipVision
        )
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/clip-vit-base-patch32"})
      serving = Bumblebee.Vision.image_embedding(clip, featurizer)
      image = StbImage.read_file!(path)
      Nx.Serving.run(serving, image)
      #=> %{
      #=>   embedding: #Nx.Tensor<
      #=>     f32[768]
      #=>     [-0.43403682112693787, 0.09786412119865417, -0.7233262062072754, -0.7707743644714355, 0.5550824403762817, -0.8923342227935791, 0.2687447965145111, 0.9633643627166748, 0.3520320951938629, 0.43195801973342896, 2.1438512802124023, -0.6542983651161194, -1.9736307859420776, 0.1611439287662506, 0.24555791914463043, 0.16985465586185455, 0.9012499451637268, 1.0657984018325806, 1.087411642074585, -0.5864712595939636, 0.3314521908760071, 0.8396108150482178, 0.3906593322753906, 0.13463366031646729, 0.2605385184288025, -0.07457947731018066, 0.4735124707221985, -0.41367805004119873, 0.18244807422161102, 1.4741417169570923, -5.807061195373535, 0.38920706510543823, 0.057687126100063324, 0.060301072895526886, 0.9680367708206177, 0.9670255184173584, 1.3876476287841797, -0.15498873591423035, -0.969764232635498, -0.38127464056015015, 0.05450016260147095, 2.2317700386047363, -0.07926210761070251, -0.11876475065946579, -1.5408644676208496, 0.7505669593811035, 0.9280041456222534, -0.3571934103965759, -1.1390857696533203, ...]
      #=>   >
      #=> }
  """
  @spec image_embedding(
          Bumblebee.model_info(),
          Bumblebee.Featurizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate image_embedding(model_info, featurizer, opts \\ []),
    to: Bumblebee.Vision.ImageEmbedding
end
