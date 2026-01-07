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

    * `"pixel_values"` - `{num_patches, flattened_patch_size}`

      Pre-extracted image/video patches from the featurizer. The shape is
      `{num_patches, channels * temporal_patch_size * patch_size * patch_size}`.
      For a 384x384 image with default settings, this is `{576, 1536}`.

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
    # Vision input is pre-extracted patches: {num_patches, flattened_patch_size}
    # flattened_patch_size = channels * temporal_patch_size * patch_size * patch_size
    patch_size = vision_spec.patch_size
    temporal_patch_size = vision_spec.temporal_patch_size

    flattened_patch_size =
      vision_spec.num_channels * temporal_patch_size * patch_size * patch_size

    # Use 196 patches as template (14x14 grid from 224x224 image)
    num_patches = 196

    %{
      "pixel_values" => Nx.template({num_patches, flattened_patch_size}, :f32),
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

    # Note: DeepStack features are extracted by vision encoder but injection
    # into text decoder is not yet implemented. The model works correctly
    # without DeepStack - it provides multi-scale visual information as an
    # enhancement.
    # TODO: Implement deepstack injection into text decoder layers 0,1,2
    # deepstack_features = Axon.nx(vision_model, & &1.deepstack_hidden_states)

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
    # Vision inputs - pre-extracted patches from featurizer
    # Shape: {num_patches, flattened_patch_size} where
    # flattened_patch_size = channels * temporal_patch_size * patch_size * patch_size
    patch_size = spec.vision_spec.patch_size
    temporal_patch_size = spec.vision_spec.temporal_patch_size

    flattened_patch_size =
      spec.vision_spec.num_channels * temporal_patch_size * patch_size * patch_size

    vision_shape = {nil, flattened_patch_size}

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
    # mask: {batch, seq_len} - boolean mask where image tokens are
    {batch_size, seq_len, hidden_size} = Nx.shape(token_embeds)
    {_, num_visual, _} = Nx.shape(visual_embeds)

    # We need to scatter visual_embeds into positions where mask is true
    # Create indices for where to place visual embeddings
    # mask_indices gives us which positions in seq_len are image tokens

    # Convert mask to indices - find positions where mask is true
    # For each position in the sequence, if it's an image token,
    # we need to know which visual embedding to use

    # Create a cumulative sum of the mask to get visual embedding indices
    # mask: [0, 0, 1, 1, 1, 0, 0] -> cumsum: [0, 0, 1, 2, 3, 3, 3]
    # Then subtract 1 where mask is true to get 0-indexed: [-, -, 0, 1, 2, -, -]
    mask_int = Nx.as_type(mask, :s32)
    cumsum = Nx.cumulative_sum(mask_int, axis: 1)
    # visual_indices gives the index into visual_embeds for each position
    # For non-image positions, this will be garbage but we'll mask it out
    visual_indices = Nx.subtract(cumsum, 1)
    # Clamp to valid range
    visual_indices = Nx.clip(visual_indices, 0, num_visual - 1)

    # Gather visual embeddings according to indices
    # visual_indices shape: {batch, seq_len}
    # We need to gather from visual_embeds {batch, num_visual, hidden}
    # Result should be {batch, seq_len, hidden}

    # Expand indices to match hidden dimension for gathering
    # {batch, seq_len} -> {batch, seq_len, hidden}
    visual_indices_expanded = Nx.new_axis(visual_indices, -1)

    visual_indices_expanded =
      Nx.broadcast(visual_indices_expanded, {batch_size, seq_len, hidden_size})

    visual_gathered = Nx.take_along_axis(visual_embeds, visual_indices_expanded, axis: 1)

    # Expand mask for broadcasting with hidden dimension
    mask_expanded = Nx.new_axis(mask, -1)
    mask_expanded = Nx.broadcast(mask_expanded, {batch_size, seq_len, hidden_size})

    # Select: where mask is true, use visual; else use token
    Nx.select(mask_expanded, visual_gathered, token_embeds)
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

      # Qwen3-VL uses QK-norm in the text model (same as standalone Qwen3)
      text_spec =
        Bumblebee.configure(Bumblebee.Text.Qwen3,
          architecture: :for_causal_language_modeling
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

      # Qwen3-VL text model uses `model.language_model.*` paths instead of Qwen3's `model.*`
      # The loader infers a "model." prefix from PyTorch state, so we use "language_model.*"
      # paths (the loader will prepend "model." automatically)
      text_mapping = %{
        "text_model.embedder.token_embedding" => "language_model.embed_tokens",
        "text_model.decoder.blocks.{n}.self_attention.query" =>
          "language_model.layers.{n}.self_attn.q_proj",
        "text_model.decoder.blocks.{n}.self_attention.key" =>
          "language_model.layers.{n}.self_attn.k_proj",
        "text_model.decoder.blocks.{n}.self_attention.value" =>
          "language_model.layers.{n}.self_attn.v_proj",
        "text_model.decoder.blocks.{n}.self_attention.output" =>
          "language_model.layers.{n}.self_attn.o_proj",
        "text_model.decoder.blocks.{n}.self_attention.query_norm" =>
          "language_model.layers.{n}.self_attn.q_norm",
        "text_model.decoder.blocks.{n}.self_attention.key_norm" =>
          "language_model.layers.{n}.self_attn.k_norm",
        "text_model.decoder.blocks.{n}.self_attention_norm" =>
          "language_model.layers.{n}.input_layernorm",
        "text_model.decoder.blocks.{n}.ffn.gate" => "language_model.layers.{n}.mlp.gate_proj",
        "text_model.decoder.blocks.{n}.ffn.intermediate" =>
          "language_model.layers.{n}.mlp.up_proj",
        "text_model.decoder.blocks.{n}.ffn.output" => "language_model.layers.{n}.mlp.down_proj",
        "text_model.decoder.blocks.{n}.output_norm" =>
          "language_model.layers.{n}.post_attention_layernorm",
        "text_model.output_norm" => "language_model.norm",
        "text_model.language_modeling_head.output" =>
          if(spec.text_spec.tie_word_embeddings,
            do: "language_model.embed_tokens",
            else: "language_model.lm_head"
          )
      }

      Map.merge(vision_mapping, text_mapping)
    end
  end
end
