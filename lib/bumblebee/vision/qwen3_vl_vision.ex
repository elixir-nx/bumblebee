defmodule Bumblebee.Vision.Qwen3VLVision do
  alias Bumblebee.Shared

  options =
    [
      hidden_size: [
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 4096,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 16,
        doc: "the size of the patch spatial dimensions"
      ],
      temporal_patch_size: [
        default: 2,
        doc: "the size of the patch temporal dimension (for video)"
      ],
      spatial_merge_size: [
        default: 2,
        doc: "the factor by which to merge spatial patches"
      ],
      out_hidden_size: [
        default: 2048,
        doc: "the output dimensionality after patch merger"
      ],
      num_position_embeddings: [
        default: 2304,
        doc: "the number of position embeddings"
      ],
      deepstack_visual_indexes: [
        default: [5, 11, 17],
        doc: "the encoder layer indices from which to extract DeepStack features (1-indexed)"
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ]

  @moduledoc """
  The Qwen3-VL vision encoder for processing images and video frames.

  ## Architectures

    * `:base` - the base vision encoder model

  ## Inputs

    * `"pixel_values"` - `{batch_size, num_channels, temporal, height, width}`

      Featurized image/video pixel values. For images, temporal=1.

    * `"grid_thw"` - `{batch_size, 3}`

      Grid dimensions [temporal, height, width] for each sample in the batch.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(spec) do
    # Template for a single image (temporal=1)
    %{
      "pixel_values" => Nx.template({1, spec.num_channels, 1, 224, 224}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    # pixel_values shape: {batch, channels, temporal, height, width}
    pixel_shape = {nil, spec.num_channels, nil, nil, nil}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: pixel_shape)
    ])
  end

  defp core(inputs, spec) do
    pixel_values = inputs["pixel_values"]

    # Patch embedding: 3D conv simulated via reshape + 2D conv + reshape
    embeddings = patch_embedding(pixel_values, spec, name: "patch_embed")

    # Note: Qwen2VL uses rotary position embeddings in attention, not learned position embeddings
    # So we skip adding position embeddings here

    # Encoder with transformer blocks
    encoder_outputs =
      encoder(embeddings, spec, name: "blocks")

    # Patch merger
    hidden_state =
      patch_merger(encoder_outputs.hidden_state, spec, name: "merger")

    %{
      hidden_state: hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      # DeepStack features from intermediate layers
      deepstack_hidden_states: encoder_outputs.deepstack_hidden_states
    }
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    # Input: {batch, channels, temporal, height, width}
    # We need to simulate 3D conv with 2D conv
    # For temporal_patch_size=2, we group pairs of frames

    # Reshape to combine temporal and batch for 2D processing
    # Then use conv with appropriate stride

    pixel_values
    |> Axon.nx(fn x ->
      # x shape: {batch, channels, temporal, height, width}
      {batch, channels, temporal, height, width} = Nx.shape(x)

      # Reshape: merge temporal into batch for 2D conv processing
      # {batch * temporal, channels, height, width}
      x = Nx.reshape(x, {batch * temporal, channels, height, width})

      # Transpose to NHWC for Axon conv
      Nx.transpose(x, axes: [0, 2, 3, 1])
    end)
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      use_bias: false,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "proj")
    )
    |> Axon.nx(fn x ->
      # x shape: {batch * temporal, h_patches, w_patches, hidden_size}
      # Reshape to {batch, num_patches, hidden_size}
      # Note: This is a simplification - the actual implementation
      # handles variable temporal dimensions more carefully
      {_bt, h, w, c} = Nx.shape(x)
      Nx.reshape(x, {:auto, h * w, c})
    end)
  end

  defp encoder(embeddings, spec, opts) do
    name = opts[:name]

    # Convert deepstack indexes to 0-indexed
    deepstack_indexes =
      spec.deepstack_visual_indexes
      |> Enum.map(&(&1 - 1))
      |> MapSet.new()

    # Use Layers.Transformer.blocks/2 as required by best practices
    # The vision encoder uses norm-first blocks without causal masking
    Layers.Transformer.blocks(embeddings,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: 0.0,
      attention_dropout_rate: 0.0,
      layer_norm: [
        epsilon: spec.layer_norm_epsilon
      ],
      ffn: [
        intermediate_size: spec.intermediate_size,
        activation: spec.activation
      ],
      block_type: :norm_first,
      # Vision encoder uses rotary embeddings
      # For now, we'll add this later when we have position_ids
      name: name
    )
    |> then(fn outputs ->
      # Extract deepstack hidden states from the collected hidden_states
      # This is done post-hoc since Layers.Transformer.blocks collects all hidden states
      deepstack_hidden_states =
        Axon.nx(outputs.hidden_states, fn hidden_states_tuple ->
          # hidden_states_tuple is a tuple of all hidden states
          # Extract the ones at deepstack_indexes
          hidden_states_list = Tuple.to_list(hidden_states_tuple)

          deepstack_indexes
          |> Enum.sort()
          |> Enum.map(fn idx ->
            if idx < length(hidden_states_list) do
              Enum.at(hidden_states_list, idx)
            else
              # Fallback to last hidden state
              List.last(hidden_states_list)
            end
          end)
          |> List.to_tuple()
        end)

      Map.put(outputs, :deepstack_hidden_states, deepstack_hidden_states)
    end)
  end

  defp patch_merger(hidden_state, spec, opts) do
    name = opts[:name]

    # Patch merger: layer norm -> spatial merge -> MLP projection
    # Note: Layer norm is applied BEFORE spatial merge in Qwen2VL
    merge_size = spec.spatial_merge_size * spec.spatial_merge_size
    mlp_input_size = spec.hidden_size * merge_size

    hidden_state
    # Layer norm on hidden_size (before merging)
    |> Axon.layer_norm(
      epsilon: spec.layer_norm_epsilon,
      name: join(name, "ln_q")
    )
    # Reshape to group spatial patches for merging
    |> Axon.nx(fn x ->
      {batch, num_patches, hidden} = Nx.shape(x)
      # Compute grid dimensions (assuming square grid)
      grid_size = :math.sqrt(num_patches) |> trunc()
      merged_grid = div(grid_size, spec.spatial_merge_size)

      # Reshape and merge spatial patches
      x
      |> Nx.reshape(
        {batch, merged_grid, spec.spatial_merge_size, merged_grid, spec.spatial_merge_size,
         hidden}
      )
      |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
      |> Nx.reshape({batch, merged_grid * merged_grid, merge_size * hidden})
    end)
    # MLP: fc1 -> activation -> fc2
    |> Axon.dense(mlp_input_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "mlp.0")
    )
    |> Layers.activation(spec.activation)
    |> Axon.dense(spec.out_hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "mlp.2")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    # Support loading from the entire Qwen3VL/Qwen2VL configuration
    def load(spec, %{"model_type" => "qwen3_vl", "vision_config" => data}) do
      load(spec, data)
    end

    def load(spec, %{"model_type" => "qwen2_vl", "vision_config" => data}) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

      # Vision config uses embed_dim for hidden_size
      opts =
        convert!(data,
          hidden_size: {"embed_dim", number()},
          num_blocks: {"depth", number()},
          num_attention_heads: {"num_heads", number()},
          num_channels: {"in_channels", number()},
          patch_size: {"patch_size", number()},
          temporal_patch_size: {"temporal_patch_size", number()},
          spatial_merge_size: {"spatial_merge_size", number()},
          activation: {"hidden_act", activation()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      # Compute derived values
      # intermediate_size = hidden_size * mlp_ratio (default mlp_ratio = 4)
      mlp_ratio = Map.get(data, "mlp_ratio", 4)
      hidden_size = opts[:hidden_size] || spec.hidden_size
      intermediate_size = hidden_size * mlp_ratio

      # out_hidden_size is typically the text model's hidden_size
      # If not specified, it comes from the parent config or defaults
      out_hidden_size = Map.get(data, "out_hidden_size", spec.out_hidden_size)

      opts =
        opts
        |> Keyword.put(:intermediate_size, intermediate_size)
        |> Keyword.put(:out_hidden_size, out_hidden_size)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        # Patch embedding - convert 3D conv kernel to 2D
        # PyTorch 3D conv shape: {out_channels, in_channels, temporal, h, w} = {32, 3, 2, 8, 8}
        # Axon 2D conv shape: {h, w, in_channels, out_channels} = {8, 8, 3, 32}
        "patch_embed.proj" => %{
          "kernel" => {
            [{"visual.patch_embed.proj", "weight"}],
            fn [kernel] ->
              # kernel shape: {out_channels, in_channels, temporal, h, w}
              # 1. Average over temporal dimension (axis 2): {out, in, t, h, w} -> {out, in, h, w}
              kernel = Nx.mean(kernel, axes: [2])
              # 2. Transpose to Axon format: {out, in, h, w} -> {h, w, in, out}
              Nx.transpose(kernel, axes: [2, 3, 1, 0])
            end
          }
        },
        # Transformer blocks
        "blocks.{n}.self_attention_norm" => "visual.blocks.{n}.norm1",
        "blocks.{n}.self_attention.query" =>
          Shared.sliced_dense_params_source(
            "visual.blocks.{n}.attn.qkv",
            {[1, 1, 1], :auto},
            0
          ),
        "blocks.{n}.self_attention.key" =>
          Shared.sliced_dense_params_source(
            "visual.blocks.{n}.attn.qkv",
            {[1, 1, 1], :auto},
            1
          ),
        "blocks.{n}.self_attention.value" =>
          Shared.sliced_dense_params_source(
            "visual.blocks.{n}.attn.qkv",
            {[1, 1, 1], :auto},
            2
          ),
        "blocks.{n}.self_attention.output" => "visual.blocks.{n}.attn.proj",
        "blocks.{n}.output_norm" => "visual.blocks.{n}.norm2",
        "blocks.{n}.ffn.intermediate" => "visual.blocks.{n}.mlp.fc1",
        "blocks.{n}.ffn.output" => "visual.blocks.{n}.mlp.fc2",
        # Patch merger
        "merger.ln_q" => "visual.merger.ln_q",
        "merger.mlp.0" => "visual.merger.mlp.0",
        "merger.mlp.2" => "visual.merger.mlp.2"
      }
    end
  end
end
