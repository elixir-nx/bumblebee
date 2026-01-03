defmodule Bumblebee.Multimodal.Siglip do
  alias Bumblebee.Shared

  options =
    [
      text_spec: [
        default: nil,
        doc: "the specification of the text model. See `Bumblebee.Text.SiglipText` for details"
      ],
      vision_spec: [
        default: nil,
        doc:
          "the specification of the vision model. See `Bumblebee.Vision.SiglipVision` for details"
      ],
      logit_scale_initial_value: [
        default: 2.6592,
        doc: "the initial value for the scaling layer used to scale similarity logits"
      ],
      logit_bias_initial_value: [
        default: -10.0,
        doc: "the initial value for the bias added to similarity logits"
      ]
    ]

  @moduledoc """
  The SigLIP model for text-image similarity.

  SigLIP uses a sigmoid loss function instead of the contrastive loss used
  by CLIP, which allows for better scaling and more stable training.

  ## Architectures

    * `:base` - the base SigLIP model

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

    #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(%{vision_spec: vision_spec}) do
    vision_shape = {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    %{
      "input_ids" => Nx.template({1, 1}, :u32),
      "pixel_values" => Nx.template(vision_shape, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    %{text_spec: text_spec, vision_spec: vision_spec} = spec

    text_shape = {nil, nil}
    vision_shape = {nil, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", shape: text_shape),
        Axon.input("attention_mask", optional: true, shape: text_shape),
        Axon.input("position_ids", optional: true, shape: text_shape),
        Axon.input("pixel_values", shape: vision_shape)
      ])

    text_model =
      text_spec
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("text_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "input_ids" => inputs["input_ids"],
        "attention_mask" => inputs["attention_mask"],
        "position_ids" => inputs["position_ids"]
      })

    vision_model =
      vision_spec
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("vision_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "pixel_values" => inputs["pixel_values"]
      })

    text_embedding =
      text_model
      |> Axon.nx(& &1.pooled_state)
      |> Axon.nx(&normalize/1)

    image_embedding =
      vision_model
      |> Axon.nx(& &1.pooled_state)
      |> Axon.nx(&normalize/1)

    similarity = Layers.cosine_similarity(text_embedding, image_embedding)

    logits_per_text =
      similarity
      |> scale_layer(
        name: "logit_scale",
        initializer: Axon.Initializers.full(spec.logit_scale_initial_value)
      )
      |> bias_layer(
        name: "logit_bias",
        initializer: Axon.Initializers.full(spec.logit_bias_initial_value)
      )

    logits_per_image = Axon.transpose(logits_per_text)

    Layers.output(%{
      logits_per_text: logits_per_text,
      logits_per_image: logits_per_image,
      text_embedding: text_embedding,
      image_embedding: image_embedding
    })
  end

  defp normalize(x) do
    Nx.divide(x, Nx.LinAlg.norm(x, ord: 2, axes: [-1], keep_axes: true))
  end

  defp scale_layer(input, opts) do
    name = opts[:name]
    initializer = opts[:initializer] || Axon.Initializers.full(1.0)

    scale_param = Axon.param("scale", fn _ -> {} end, initializer: initializer)

    Axon.layer(
      fn input, scale, _opts ->
        Nx.multiply(input, Nx.exp(scale))
      end,
      [input, scale_param],
      name: name,
      op_name: :logit_scale
    )
  end

  defp bias_layer(input, opts) do
    name = opts[:name]
    initializer = opts[:initializer] || Axon.Initializers.full(0.0)

    bias_param = Axon.param("bias", fn _ -> {} end, initializer: initializer)

    Axon.layer(
      fn input, bias, _opts ->
        Nx.add(input, bias)
      end,
      [input, bias_param],
      name: name,
      op_name: :logit_bias
    )
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      {text_data, data} = Map.pop(data, "text_config", %{})
      {vision_data, data} = Map.pop(data, "vision_config", %{})

      text_spec =
        Bumblebee.Text.SiglipText
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(text_data)

      vision_spec =
        Bumblebee.Vision.SiglipVision
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(vision_data)

      opts =
        convert!(data,
          logit_scale_initial_value: {"logit_scale_init_value", number()},
          logit_bias_initial_value: {"logit_bias_init_value", number()}
        )

      @for.config(spec, opts ++ [text_spec: text_spec, vision_spec: vision_spec])
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.HuggingFace.Transformers

    def params_mapping(spec) do
      text_mapping =
        spec.text_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("text_model", nil)

      vision_mapping =
        spec.vision_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("vision_model", nil)

      %{
        "logit_scale" => %{
          "scale" => {[{"logit_scale", "logit_scale"}], fn [scale] -> Nx.squeeze(scale) end}
        },
        "logit_bias" => %{
          "bias" => {[{"logit_bias", "logit_bias"}], fn [bias] -> Nx.squeeze(bias) end}
        }
      }
      |> Map.merge(text_mapping)
      |> Map.merge(vision_mapping)
    end
  end
end
