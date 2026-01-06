defmodule Bumblebee.Multimodal.Qwen3VL do
  alias Bumblebee.Shared

  options =
    [
      image_token_id: [
        default: 151_655,
        doc: "the token ID used to represent images in the input sequence"
      ],
      video_token_id: [
        default: 151_656,
        doc: "the token ID used to represent videos in the input sequence"
      ],
      vision_start_token_id: [
        default: 151_652,
        doc: "the token ID marking the start of visual content"
      ],
      vision_end_token_id: [
        default: 151_653,
        doc: "the token ID marking the end of visual content"
      ]
    ] ++ Shared.common_options([:output_hidden_states, :output_attentions])

  @moduledoc """
  Qwen3-VL model for vision-language tasks.

  ## Architectures

    * `:for_conditional_generation` - Qwen3-VL with a language modeling
      head for image/video-to-text generation

  ## Inputs

    * `"pixel_values"` - `{batch_size, num_channels, temporal, height, width}`

      Featurized image/video pixel values. For images, temporal=1.

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary. Should contain
      special image/video tokens at positions where visual content appears.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

  """

  defstruct [architecture: :for_conditional_generation, vision_spec: nil, text_spec: nil] ++
              Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:for_conditional_generation]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(%{vision_spec: vision_spec}) do
    %{
      # Vision input: {batch, channels, temporal, height, width}
      "pixel_values" => Nx.template({1, vision_spec.num_channels, 1, 224, 224}, :f32),
      "input_ids" => Nx.template({1, 1}, :u32)
    }
  end

  @impl true
  def init_cache(%{text_spec: text_spec}, batch_size, max_length, inputs) do
    text_spec.__struct__.init_cache(text_spec, batch_size, max_length, inputs)
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  @impl true
  def model(%__MODULE__{architecture: :for_conditional_generation} = spec) do
    inputs = inputs(spec)

    vision_model =
      Bumblebee.build_model(spec.vision_spec)
      |> Bumblebee.Utils.Axon.prefix_names("vision_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "pixel_values" => inputs["pixel_values"]
      })

    # Get vision embeddings using correct Axon.nx pattern
    vision_hidden_state =
      Layers.if_present inputs["pixel_values"] do
        Axon.nx(vision_model, & &1.hidden_state)
      else
        Layers.none()
      end

    # Build text model
    text_model =
      Bumblebee.build_model(spec.text_spec)
      |> Bumblebee.Utils.Axon.prefix_names("text_model.")

    # Substitute visual embeddings into text input
    input_embeddings =
      substitute_visual_embeddings(
        inputs["input_ids"],
        vision_hidden_state,
        spec,
        name: "embed_substitute"
      )

    # Run text model with substituted embeddings
    text_outputs =
      text_model
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "input_embeddings" => input_embeddings,
        "attention_mask" => inputs["attention_mask"],
        "position_ids" => inputs["position_ids"],
        "cache" => inputs["cache"]
      })

    Layers.output(%{
      logits: Axon.nx(text_outputs, & &1.logits),
      cache: Axon.nx(text_outputs, & &1.cache),
      hidden_states: Axon.nx(text_outputs, & &1.hidden_states),
      attentions: Axon.nx(text_outputs, & &1.attentions)
    })
  end

  defp inputs(spec) do
    # Vision inputs
    vision_shape = {nil, spec.vision_spec.num_channels, nil, nil, nil}

    # Text inputs
    text_shape = {nil, nil}
    hidden_shape = {nil, nil, spec.text_spec.hidden_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", optional: true, shape: vision_shape),
      Axon.input("input_ids", shape: text_shape),
      Axon.input("attention_mask", optional: true, shape: text_shape),
      Axon.input("position_ids", optional: true, shape: text_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp substitute_visual_embeddings(input_ids, vision_hidden_state, spec, _opts) do
    # Get the token embeddings for the input_ids
    token_embeddings =
      Axon.embedding(input_ids, spec.text_spec.vocab_size, spec.text_spec.hidden_size,
        name: "text_model.embedder.token_embedding"
      )

    # If no vision input, just return token embeddings
    # Otherwise, substitute visual embeddings at image/video token positions
    Layers.if_present vision_hidden_state do
      Axon.layer(
        fn token_embeds, visual_embeds, input_ids, _opts ->
          # Create mask for visual tokens
          image_mask = Nx.equal(input_ids, spec.image_token_id)
          video_mask = Nx.equal(input_ids, spec.video_token_id)
          visual_mask = Nx.logical_or(image_mask, video_mask)

          # visual_embeds shape: {batch, num_visual_tokens, hidden_size}
          # visual_mask shape: {batch, seq_len}
          # This is a simplified substitution - a full implementation would need
          # to handle variable numbers of visual tokens per sequence
          substitute_at_mask(token_embeds, visual_embeds, visual_mask)
        end,
        [token_embeddings, vision_hidden_state, input_ids]
      )
    else
      # No visual input - just use token embeddings
      token_embeddings
    end
  end

  # Substitute visual embeddings at positions where mask is true
  defp substitute_at_mask(token_embeds, visual_embeds, mask) do
    # token_embeds: {batch, seq_len, hidden}
    # visual_embeds: {batch, num_visual, hidden}
    # mask: {batch, seq_len} - boolean mask
    {batch_size, seq_len, hidden_size} = Nx.shape(token_embeds)
    {_, num_visual, _} = Nx.shape(visual_embeds)

    # For each batch, find the positions where mask is true and substitute
    # This is a simplified version - we assume visual tokens are contiguous
    # and in the same order as visual_embeds

    # Expand mask for broadcasting
    mask_expanded = Nx.new_axis(mask, -1)
    mask_expanded = Nx.broadcast(mask_expanded, {batch_size, seq_len, hidden_size})

    # Pad or truncate visual_embeds to match seq_len
    visual_padded =
      if num_visual < seq_len do
        # Pad with zeros
        padding = Nx.broadcast(0.0, {batch_size, seq_len - num_visual, hidden_size})
        Nx.concatenate([visual_embeds, padding], axis: 1)
      else
        # Truncate
        Nx.slice(visual_embeds, [0, 0, 0], [batch_size, seq_len, hidden_size])
      end

    # Use scatter-like operation: where mask is true, use visual; else use token
    Nx.select(mask_expanded, visual_padded, token_embeds)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          image_token_id: {"image_token_id", number()},
          video_token_id: {"video_token_id", number()},
          vision_start_token_id: {"vision_start_token_id", number()},
          vision_end_token_id: {"vision_end_token_id", number()}
        )

      # Load text spec from text_config first to get hidden_size
      text_data = Map.get(data, "text_config", data)

      # Qwen2VL doesn't use QK-norm in the text model (unlike standalone Qwen3)
      text_spec =
        Bumblebee.configure(Bumblebee.Text.Qwen3,
          architecture: :for_causal_language_modeling,
          use_qk_norm: false
        )
        |> Bumblebee.HuggingFace.Transformers.Config.load(text_data)

      # Load vision spec with out_hidden_size from text config
      vision_data =
        data
        |> Map.put_new("vision_config", %{})
        |> update_in(["vision_config"], fn vc ->
          Map.put_new(vc, "out_hidden_size", text_spec.hidden_size)
        end)

      vision_spec =
        Bumblebee.configure(Bumblebee.Vision.Qwen3VLVision)
        |> Bumblebee.HuggingFace.Transformers.Config.load(vision_data)

      @for.config(
        %{spec | vision_spec: vision_spec, text_spec: text_spec},
        opts
      )
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      vision_mapping =
        Bumblebee.HuggingFace.Transformers.Model.params_mapping(spec.vision_spec)
        |> Enum.map(fn {bumblebee, hf} -> {"vision_model.#{bumblebee}", hf} end)
        |> Map.new()

      text_mapping =
        Bumblebee.HuggingFace.Transformers.Model.params_mapping(spec.text_spec)
        |> Enum.map(fn {bumblebee, hf} -> {"text_model.#{bumblebee}", hf} end)
        |> Map.new()

      Map.merge(vision_mapping, text_mapping)
    end
  end
end
