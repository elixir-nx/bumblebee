defmodule Bumblebee.Vision do
  @moduledoc """
  High-level tasks related to vision.
  """

  @typedoc """
  A term representing an image.

  Either `Nx.Tensor` in HWC order or a struct implementing `Nx.Container`
  and resolving to such tensor.
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

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

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
end
