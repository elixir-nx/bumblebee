defmodule Bumblebee.Diffusion.StableDiffusion.SafetyChecker do
  alias Bumblebee.Shared

  options = [
    clip_spec: [
      default: nil,
      doc: "the specification of the CLIP model. See `Bumblebee.Multimodal.Clip` for details"
    ]
  ]

  @moduledoc """
  A CLIP-based model for detecting unsafe image content.

  This model is designed primarily to check images generated using
  Stable Diffusion.

  ## Architectures

    * `:base` - the base safety detection model

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4#safety-module)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Nx.Defn

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(%{clip_spec: %{vision_spec: vision_spec}}) do
    vision_shape = {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    %{"pixel_values" => Nx.template(vision_shape, :f32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    %{clip_spec: %{vision_spec: vision_spec}} = spec

    vision_shape = {nil, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("pixel_values", shape: vision_shape)
      ])

    vision_model =
      vision_spec
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("vision_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "pixel_values" => inputs["pixel_values"]
      })

    image_embeddings =
      vision_model
      |> Axon.nx(& &1.pooled_state)
      |> Axon.dense(spec.clip_spec.projection_size, use_bias: false, name: "visual_projection")

    is_unsafe = unsafe_detection(image_embeddings, spec, name: "unsafe_detection")

    Layers.output(%{
      is_unsafe: is_unsafe
    })
  end

  defp unsafe_detection(image_embeddings, spec, opts) do
    name = opts[:name]

    # The embeddings are precomputed using the CLIP text model and
    # represent sensitive/unsafe concepts in the latent space. We then
    # check whether an image is far enough from those concepts in the
    # latent space (using a hand-engineered threshold for each concept).

    num_sensitive_concepts = 3
    num_unsafe_concepts = 17

    sensitive_concept_embeddings =
      Axon.param("sensitive_concept_embeddings", fn _ ->
        {num_sensitive_concepts, spec.clip_spec.projection_size}
      end)

    unsafe_concept_embeddings =
      Axon.param("unsafe_concept_embeddings", fn _ ->
        {num_unsafe_concepts, spec.clip_spec.projection_size}
      end)

    sensitive_concept_thresholds =
      Axon.param("sensitive_concept_thresholds", fn _ -> {num_sensitive_concepts} end)

    unsafe_concept_thresholds =
      Axon.param("unsafe_concept_thresholds", fn _ -> {num_unsafe_concepts} end)

    Axon.layer(
      &unsafe_detection_impl/6,
      [
        image_embeddings,
        sensitive_concept_embeddings,
        unsafe_concept_embeddings,
        sensitive_concept_thresholds,
        unsafe_concept_thresholds
      ],
      name: name
    )
  end

  defnp unsafe_detection_impl(
          image_embeddings,
          sensitive_concept_embeddings,
          unsafe_concept_embeddings,
          sensitive_concept_thresholds,
          unsafe_concept_thresholds,
          _opts \\ []
        ) do
    sensitive_concept_distances =
      Bumblebee.Utils.Nx.cosine_similarity(image_embeddings, sensitive_concept_embeddings)

    unsafe_concept_distances =
      Bumblebee.Utils.Nx.cosine_similarity(image_embeddings, unsafe_concept_embeddings)

    sensitive_concept_thresholds = Nx.new_axis(sensitive_concept_thresholds, 0)
    unsafe_concept_thresholds = Nx.new_axis(unsafe_concept_thresholds, 0)

    sensitive_concept_scores = sensitive_concept_distances - sensitive_concept_thresholds
    sensitive? = Nx.any(sensitive_concept_scores > 0, axes: [1], keep_axes: true)

    # Use a lower threshold if an image has any sensitive concept
    unsafe_threshold_adjustment = Nx.select(sensitive?, 0.01, 0.0)

    unsafe_concept_scores =
      unsafe_concept_distances - unsafe_concept_thresholds + unsafe_threshold_adjustment

    Nx.any(unsafe_concept_scores > 0, axes: [1])
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      clip_spec =
        Bumblebee.Multimodal.Clip
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(data)

      @for.config(spec, clip_spec: clip_spec)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.HuggingFace.Transformers

    def params_mapping(spec) do
      vision_mapping =
        spec.clip_spec.vision_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("vision_model", "vision_model")

      %{
        "visual_projection" => "visual_projection",
        "unsafe_detection" => %{
          "sensitive_concept_embeddings" => {
            [{"unsafe_detection", "special_care_embeds"}],
            fn [value] -> value end
          },
          "unsafe_concept_embeddings" => {
            [{"unsafe_detection", "concept_embeds"}],
            fn [value] -> value end
          },
          "sensitive_concept_thresholds" => {
            [{"unsafe_detection", "special_care_embeds_weights"}],
            fn [value] -> value end
          },
          "unsafe_concept_thresholds" => {
            [{"unsafe_detection", "concept_embeds_weights"}],
            fn [value] -> value end
          }
        }
      }
      |> Map.merge(vision_mapping)
    end
  end
end
