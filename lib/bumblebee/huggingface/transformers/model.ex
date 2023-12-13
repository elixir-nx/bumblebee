defprotocol Bumblebee.HuggingFace.Transformers.Model do
  @moduledoc false

  # This protocol defines details related to loading Bumblebee model
  # from huggingface/transformers model.

  @type params_mapping :: %{layer_name() => params_source()}

  @type params_source :: layer_name() | list(layer_name()) | param_builders()

  @type param_builders :: %{param_name() => param_builder()}

  @type param_builder ::
          {list(param_source()), (list(Nx.tensor()) -> Nx.Tensor.t() | Nx.Container.t())}

  @type param_source :: param_ref() | list(param_ref())
  @type param_ref :: {layer_name(), param_name()}

  @type layer_name :: String.t()
  @type param_name :: String.t()

  @doc """
  Returns a map describing layers/parameters relationship between an
  Axon model and a corresponding huggingface/transformers model.

  ## Mapping format

  The basic mapping format is a map with Axon layer names (target) as
  keys and PyTorch layer names (source) as values. For example:

      %{
        "embedder.token_embedding" => "bert.embeddings.word_embeddings",
        ...
      }

  The mapping should always use longest names, that is, depending on
  the architecture, the PyTorch layer name could be either
  `"bert.embeddings.word_embeddings"` or `"embeddings.word_embeddings"`.
  The longer version should generally be used. Prefixes are removed/added
  as necessary, so loading partial models supported automatically.

  The layer names may include simple substitutions, useful for lists
  of layers:

      %{
        "encoder.blocks.{n}.self_attention.query" => "bert.encoder.layer.{n}.attention.self.query",
        ...
      }

  Both param names and values for corresponding layers may not match
  exactly, so they require further transformations. For example, the
  convolution `"kernel"` in Axon corresponds to a transposed `"weight"`
  from PyTorch. For most common layers such conversions are handled
  automatically.

  In some cases, particularly with model-specific layers/parameters,
  we may need more control over the parameter mapping. In such cases,
  instead of source layer name, a map with parameter-level transformations
  may be specified:

      %{
        "embedder.class_embedding" => %{
          "embedding" => {
            [{"vit.embeddings", "cls_token"}],
            fn [value] -> Nx.squeeze(value, axes: [0, 1]) end
          }
        },
        ...
      }

  For each parameter, we specify a list of source parameters in the
  form of `{source_layer_name, source_param_name}`, then a function
  to build our parameter value. Instead of a single tuple, we can
  specify a list of those to try one by one. With the explicit
  transformation we can handle arbitrary parameter name and value
  transformations.
  """
  @spec params_mapping(t()) :: params_mapping()
  def params_mapping(spec)
end
