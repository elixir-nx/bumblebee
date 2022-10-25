defmodule Bumblebee.Multimodal.Clip do
  alias Bumblebee.Shared

  options = [
    projection_size: [
      default: 512,
      doc: "the dimensionality of text and vision projection layers"
    ],
    logit_scale_initial_value: [
      default: 2.6592,
      doc: "the initial value for the scaling layer used to scale similarity logits"
    ]
  ]

  @moduledoc """
  The CLIP model for text-image similarity.

  ## Architectures

    * `:base` - the base CLIP model

  ## Inputs

    * `"input_ids"` - `{batch_size, seq_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, seq_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.


    * `"position_ids"` - `{batch_size, seq_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.


    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [CLIP: Connecting Text and Images](https://openai.com/blog/clip)

    * [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

  """

  alias Bumblebee.Layers

  defstruct [
              architecture: :base,
              text_spec: nil,
              vision_spec: nil
            ] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts \\ []) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(%{vision_spec: vision_spec}) do
    vision_shape = {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    %{
      "input_ids" => Nx.template({1, 1}, :s64),
      "pixel_values" => Nx.template(vision_shape, :s64)
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
      |> Bumblebee.Vision.ClipVision.model()
      |> Bumblebee.Utils.Axon.prefix_names("vision_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "pixel_values" => inputs["pixel_values"]
      })

    text_embeddings =
      text_model
      |> Axon.nx(& &1.pooler_output)
      |> Axon.dense(spec.projection_size, use_bias: false, name: "text_projection")

    image_embeddings =
      vision_model
      |> Axon.nx(& &1.pooler_output)
      |> Axon.dense(spec.projection_size, use_bias: false, name: "visual_projection")

    logits_per_text =
      text_embeddings
      |> cosine_similarity(image_embeddings)
      |> exp_scale(
        name: "scale",
        scale_name: "logit_scale",
        scale_initializer: Axon.Initializers.full(spec.logit_scale_initial_value)
      )

    logits_per_image = Axon.transpose(logits_per_text)

    Layers.output(%{
      logits_per_text: logits_per_text,
      logits_per_image: logits_per_image,
      text_embeddings: text_embeddings,
      image_embeddings: image_embeddings
    })
  end

  defp cosine_similarity(x, y) do
    Axon.layer(
      fn x, y, _opts ->
        x = normalize(x)
        y = normalize(y)
        Nx.dot(x, [1], y, [1])
      end,
      [x, y],
      op_names: :cosine_similarity
    )
  end

  defp normalize(tensor) do
    norm =
      tensor
      |> Nx.power(2)
      |> Nx.sum(axes: [-1], keep_axes: true)
      |> Nx.sqrt()

    Nx.divide(tensor, norm)
  end

  defp exp_scale(input, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        scale_name: "scale",
        scale_initializer: Axon.Initializers.full(1.0e-6)
      ])

    name = opts[:name]
    scale_name = opts[:scale_name]
    scale_initializer = opts[:scale_initializer]

    scale_param = Axon.param(scale_name, fn _ -> {} end, initializer: scale_initializer)

    Axon.layer(
      fn input, scale, _opts ->
        Nx.multiply(input, Nx.exp(scale))
      end,
      [input, scale_param],
      name: name,
      op_name: :exp_scale
    )
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      {text_data, data} = Map.pop(data, "text_config", %{})
      {vision_data, data} = Map.pop(data, "vision_config", %{})

      text_spec =
        Bumblebee.Text.ClipText
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(text_data)

      vision_spec =
        Bumblebee.Vision.ClipVision
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(vision_data)

      opts =
        convert!(data,
          projection_size: {"projection_dim", number()},
          logit_scale_initial_value: {"logit_scale_init_value", number()}
        )

      @for.config(spec, opts ++ [text_spec: text_spec, vision_spec: vision_spec])
    end
  end
end
