defmodule Bumblebee.Multimodal.Mistral3 do
  alias Bumblebee.Shared

  options =
    [
      text_spec: [
        default: nil,
        doc: "the specification of the text model. See `Bumblebee.Text.Mistral3` for details"
      ],
      vision_spec: [
        default: nil,
        doc: "the specification of the vision model. See `Bumblebee.Vision.Pixtral` for details"
      ],
      image_token_index: [
        default: 10,
        doc: "the token index used to represent image embeddings in the vocabulary"
      ],
      spatial_merge_size: [
        default: 2,
        doc: "factor by which to reduce spatial dimensions of vision features"
      ],
      projector_hidden_act: [
        default: :gelu,
        doc: "the activation function for the multimodal projector"
      ],
      vision_feature_layer: [
        default: -1,
        doc:
          "the layer index to extract vision features from (requires output_hidden_states: true for values other than -1)"
      ]
    ]

  @moduledoc """
  Mistral 3 multimodal model for vision-language understanding.

  This model combines a Pixtral vision encoder with a Mistral3 text decoder
  for multimodal tasks like image captioning and visual question answering.

  ## Architectures

    * `:for_conditional_generation` - Mistral3 multimodal model with a language
      modeling head for generating text conditioned on images

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary, including special
      image tokens that will be replaced with vision features.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"encoder_hidden_state"` - `{batch_size, num_patches, hidden_size}`

      Pre-computed vision features. If specified, the model will skip the
      image encoding process and use this value directly.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression).

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :for_conditional_generation] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:for_conditional_generation]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(%{vision_spec: vision_spec}) do
    vision_shape = {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    %{
      "pixel_values" => Nx.template(vision_shape, :f32),
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :for_conditional_generation} = spec) do
    %{vision_spec: vision_spec, text_spec: text_spec} = spec

    vision_shape = {nil, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}
    text_shape = {nil, nil}
    vision_hidden_shape = {nil, nil, vision_spec.hidden_size}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("pixel_values", optional: true, shape: vision_shape),
        Axon.input("input_ids", shape: text_shape),
        Axon.input("attention_mask", optional: true, shape: text_shape),
        Axon.input("position_ids", optional: true, shape: text_shape),
        Axon.input("encoder_hidden_state", optional: true, shape: vision_hidden_shape),
        Axon.input("cache", optional: true)
      ])

    # Build vision encoder
    vision_model =
      vision_spec
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("vision_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "pixel_values" => inputs["pixel_values"]
      })

    # Get vision features (either from encoder or pre-computed)
    # Use vision_feature_layer to select which layer's output to use
    # -1 means use the final hidden state (most common case), other values
    # select from hidden_states tuple (requires output_hidden_states: true)
    vision_features =
      Layers.if_present inputs["encoder_hidden_state"] do
        inputs["encoder_hidden_state"]
      else
        Layers.if_present inputs["pixel_values"] do
          if spec.vision_feature_layer == -1 do
            # Default case: use the final normalized hidden state directly
            Axon.nx(vision_model, & &1.hidden_state)
          else
            # Use intermediate layer: extract from hidden_states tuple
            extract_vision_features(vision_model, spec.vision_feature_layer)
          end
        else
          Layers.none()
        end
      end

    # Project vision features to text embedding space
    projected_vision_features =
      Layers.if_present vision_features do
        multimodal_projector(vision_features, spec, name: "multimodal_projector")
      else
        Layers.none()
      end

    # Get text embeddings
    text_embeddings =
      Axon.embedding(inputs["input_ids"], text_spec.vocab_size, text_spec.hidden_size,
        kernel_initializer: Axon.Initializers.normal(scale: text_spec.initializer_scale),
        name: "language_model.model.embed_tokens"
      )

    # Merge vision and text embeddings
    merged_embeddings =
      Layers.if_present projected_vision_features do
        merge_vision_text_embeddings(
          text_embeddings,
          projected_vision_features,
          inputs["input_ids"],
          spec.image_token_index
        )
      else
        text_embeddings
      end

    # Build text decoder using merged embeddings
    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(merged_embeddings)
      end

    decoder_outputs =
      decoder(
        merged_embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["cache"],
        text_spec,
        name: "language_model.model"
      )

    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "language_model.model.norm",
        epsilon: text_spec.layer_norm_epsilon
      )

    # Language modeling head
    logits =
      Layers.dense_transposed(hidden_state, text_spec.vocab_size,
        kernel_initializer: Axon.Initializers.normal(scale: text_spec.initializer_scale),
        name: "language_model.lm_head"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: decoder_outputs.hidden_states,
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache,
      vision_features: vision_features
    })
  end

  defp extract_vision_features(vision_model, vision_feature_layer) do
    # Extract vision features from the appropriate layer
    # vision_feature_layer: -1 means final layer, other values index into hidden_states
    Axon.layer(
      fn vision_output, opts ->
        unless opts[:output_hidden_states] do
          raise ArgumentError,
                "vision_feature_layer requires output_hidden_states: true. " <>
                  "Build the model with global_layer_options: [output_hidden_states: true]"
        end

        hidden_states = vision_output.hidden_states

        if match?(%Axon.None{}, hidden_states) do
          raise ArgumentError,
                "vision_feature_layer requires hidden_states from the vision encoder. " <>
                  "Make sure output_hidden_states is enabled"
        end

        # hidden_states is a tuple with embeddings + all block outputs
        # Index 0 is initial embeddings, indices 1..num_blocks are block outputs
        num_states = tuple_size(hidden_states)

        # Handle negative indexing (Python-style): -1 means last element
        layer_idx =
          if vision_feature_layer < 0 do
            num_states + vision_feature_layer
          else
            vision_feature_layer
          end

        # Clamp to valid range
        layer_idx = max(0, min(layer_idx, num_states - 1))

        elem(hidden_states, layer_idx)
      end,
      [vision_model],
      op_name: :extract_vision_features,
      global_options: [:output_hidden_states]
    )
  end

  defp multimodal_projector(vision_features, spec, opts) do
    name = opts[:name]

    # Mistral3 multimodal projector structure:
    # 1. Patch merger: merge spatial_merge_size^2 patches into one
    # 2. Norm: RMSNorm on vision features
    # 3. Linear1: vision_hidden -> text_hidden
    # 4. Linear2: text_hidden -> text_hidden
    vision_features
    |> patch_merger(spec, name: join(name, "patch_merger"))
    |> Layers.rms_norm(name: join(name, "norm"), epsilon: 1.0e-5)
    |> Axon.dense(spec.text_spec.hidden_size,
      use_bias: false,
      name: join(name, "linear_1")
    )
    |> Layers.activation(spec.projector_hidden_act)
    |> Axon.dense(spec.text_spec.hidden_size,
      use_bias: false,
      name: join(name, "linear_2")
    )
  end

  defp patch_merger(vision_features, spec, opts) do
    name = opts[:name]
    merge_size = spec.spatial_merge_size

    # The patch merger reshapes and concatenates spatial_merge_size^2 patches
    # then projects them down to the original vision hidden size
    # Input: {batch, num_patches, vision_hidden}
    # After reshape: {batch, num_merged_patches, vision_hidden * merge_size^2}
    # After linear: {batch, num_merged_patches, vision_hidden}
    merged_features =
      Axon.layer(
        fn features, _opts ->
          {batch, num_patches, hidden} = Nx.shape(features)

          # Calculate merged dimensions
          # Assume patches are arranged in a square grid
          patches_per_side = trunc(:math.sqrt(num_patches))
          merged_per_side = div(patches_per_side, merge_size)
          num_merged = merged_per_side * merged_per_side

          # Reshape to group patches for merging
          # {batch, patches_per_side, patches_per_side, hidden}
          features = Nx.reshape(features, {batch, patches_per_side, patches_per_side, hidden})

          # Reshape to {batch, merged_per_side, merge_size, merged_per_side, merge_size, hidden}
          features =
            Nx.reshape(features, {batch, merged_per_side, merge_size, merged_per_side, merge_size, hidden})

          # Transpose to group merge_size patches together
          # {batch, merged_per_side, merged_per_side, merge_size, merge_size, hidden}
          features = Nx.transpose(features, axes: [0, 1, 3, 2, 4, 5])

          # Reshape to concatenate merged patches
          # {batch, num_merged, merge_size^2 * hidden}
          Nx.reshape(features, {batch, num_merged, merge_size * merge_size * hidden})
        end,
        [vision_features]
      )

    Axon.dense(merged_features, spec.vision_spec.hidden_size,
      use_bias: false,
      name: join(name, "merging_layer")
    )
  end

  defp merge_vision_text_embeddings(
         text_embeddings,
         vision_features,
         input_ids,
         image_token_index
       ) do
    # Replace image token embeddings with projected vision features
    Axon.layer(
      fn text_emb, vision_feat, ids, _opts ->
        # Find positions where image tokens are
        # image_mask shape: {batch_size, seq_len}
        image_mask = Nx.equal(ids, image_token_index)

        # Calculate patch indices using cumulative sum
        # For each image token position, determine which patch it should get
        # First image token -> patch 0, second -> patch 1, etc.
        # cumsum shape: {batch_size, seq_len}
        cumsum = Nx.cumulative_sum(Nx.as_type(image_mask, :s32), axis: 1)

        # Convert to 0-indexed patch indices (cumsum is 1-indexed at image positions)
        # Subtract 1, but clamp to 0 for non-image positions to avoid negative indices
        patch_indices = Nx.max(Nx.subtract(cumsum, 1), 0)

        # Get dimensions
        {_batch, num_patches, _hidden_size} = Nx.shape(vision_feat)

        # Clamp patch indices to valid range to avoid out-of-bounds access
        patch_indices = Nx.min(patch_indices, num_patches - 1)

        # Expand patch_indices to match text_emb dimensions for take_along_axis
        # Use text_emb shape directly for proper dynamic shape handling
        # {batch_size, seq_len} -> {batch_size, seq_len, 1} -> {batch_size, seq_len, hidden_size}
        patch_indices_expanded =
          patch_indices
          |> Nx.new_axis(-1)
          |> Nx.broadcast(Nx.shape(text_emb))

        # Gather vision features for each position using take_along_axis
        # vision_feat: {batch_size, num_patches, hidden_size}
        # patch_indices_expanded: {batch_size, seq_len, hidden_size}
        # Result: {batch_size, seq_len, hidden_size}
        gathered_vision = Nx.take_along_axis(vision_feat, patch_indices_expanded, axis: 1)

        # Replace text embeddings with vision features only at image token positions
        # Expand image_mask to match text_emb dimensions exactly
        mask_expanded =
          image_mask
          |> Nx.new_axis(-1)
          |> Nx.broadcast(Nx.shape(text_emb))

        Nx.select(mask_expanded, gathered_vision, text_emb)
      end,
      [text_embeddings, vision_features, input_ids]
    )
  end

  defp decoder(hidden_state, position_ids, attention_mask, cache, spec, opts) do
    name = opts[:name]

    # Build attention_window_size for interleaved attention
    # If sliding_window is nil, use global attention for all layers
    attention_window_size =
      cond do
        # If no sliding window is configured, use global attention for all layers
        spec.attention_window_size == nil ->
          nil

        # Interleaved attention: even layers use global, odd layers use sliding window
        spec.use_interleaved_attention ->
          fn layer_idx ->
            if rem(layer_idx, 2) == 0 do
              nil
            else
              {spec.attention_window_size, spec.attention_window_size}
            end
          end

        # Non-interleaved: apply sliding window to all layers
        true ->
          {spec.attention_window_size, spec.attention_window_size}
      end

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      cache: cache,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      kernel_initializer: Axon.Initializers.normal(scale: spec.initializer_scale),
      layer_norm: &Layers.rms_norm(&1, name: &2, epsilon: spec.layer_norm_epsilon),
      ffn:
        &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
          name: &2,
          activation: spec.activation
        ),
      block_type: :norm_first,
      causal: true,
      attention_window_size: attention_window_size,
      rotary_embedding: [
        position_ids: position_ids,
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base
      ],
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      output_use_bias: false,
      name: join(name, "layers")
    )
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    gate = Axon.dense(hidden_state, intermediate_size, name: join(name, "gate"), use_bias: false)

    hidden_state = Axon.multiply(intermediate, Layers.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
  end

  @impl true
  def init_cache(
        %{vision_spec: _vision_spec, text_spec: text_spec},
        batch_size,
        max_length,
        _inputs
      ) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: text_spec.hidden_size,
      decoder_num_attention_heads: text_spec.num_attention_heads,
      decoder_num_blocks: text_spec.num_blocks,
      attention_head_size: text_spec.attention_head_size
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      {text_data, data} = Map.pop(data, "text_config", %{})
      {vision_data, data} = Map.pop(data, "vision_config", %{})

      # Merge top-level tie_word_embeddings into text_config if not already set
      text_data =
        if Map.has_key?(data, "tie_word_embeddings") and not Map.has_key?(text_data, "tie_word_embeddings") do
          Map.put(text_data, "tie_word_embeddings", data["tie_word_embeddings"])
        else
          text_data
        end

      text_spec =
        Bumblebee.Text.Mistral3
        |> Bumblebee.configure(architecture: :for_causal_language_modeling)
        |> Bumblebee.HuggingFace.Transformers.Config.load(text_data)

      vision_spec =
        Bumblebee.Vision.Pixtral
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(vision_data)

      opts =
        convert!(data,
          image_token_index: {"image_token_index", number()},
          spatial_merge_size: {"spatial_merge_size", number()},
          projector_hidden_act: {"projector_hidden_act", activation()},
          vision_feature_layer: {"vision_feature_layer", number()}
        )

      @for.config(spec, opts ++ [text_spec: text_spec, vision_spec: vision_spec])
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.HuggingFace.Transformers

    def params_mapping(spec) do
      vision_mapping =
        spec.vision_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("vision_model", "vision_tower")

      %{
        "language_model.model.embed_tokens" => "language_model.model.embed_tokens",
        "language_model.model.layers.{n}.self_attention.query" =>
          "language_model.model.layers.{n}.self_attn.q_proj",
        "language_model.model.layers.{n}.self_attention.key" =>
          "language_model.model.layers.{n}.self_attn.k_proj",
        "language_model.model.layers.{n}.self_attention.value" =>
          "language_model.model.layers.{n}.self_attn.v_proj",
        "language_model.model.layers.{n}.self_attention.output" =>
          "language_model.model.layers.{n}.self_attn.o_proj",
        "language_model.model.layers.{n}.self_attention_norm" =>
          "language_model.model.layers.{n}.input_layernorm",
        "language_model.model.layers.{n}.ffn.gate" =>
          "language_model.model.layers.{n}.mlp.gate_proj",
        "language_model.model.layers.{n}.ffn.intermediate" =>
          "language_model.model.layers.{n}.mlp.up_proj",
        "language_model.model.layers.{n}.ffn.output" =>
          "language_model.model.layers.{n}.mlp.down_proj",
        "language_model.model.layers.{n}.output_norm" =>
          "language_model.model.layers.{n}.post_attention_layernorm",
        "language_model.model.norm" => "language_model.model.norm",
        "language_model.lm_head" =>
          if(spec.text_spec.tie_word_embeddings,
            do: "language_model.model.embed_tokens",
            else: "language_model.lm_head"
          ),
        "multimodal_projector.patch_merger.merging_layer" =>
          "multi_modal_projector.patch_merger.merging_layer",
        "multimodal_projector.norm" => "multi_modal_projector.norm",
        "multimodal_projector.linear_1" => "multi_modal_projector.linear_1",
        "multimodal_projector.linear_2" => "multi_modal_projector.linear_2"
      }
      |> Map.merge(vision_mapping)
    end
  end
end
