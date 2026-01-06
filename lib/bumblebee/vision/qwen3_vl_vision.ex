defmodule Bumblebee.Vision.Qwen3VLVision do
  import Nx.Defn

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
    # Template for pre-extracted patches
    # For a 224x224 image: 224/16 = 14 patches per side, 14*14 = 196 patches
    # With temporal duplication (1->2), patches_t = 1
    # Total patches = 1 * 14 * 14 = 196
    patch_size = spec.patch_size
    temporal_patch_size = spec.temporal_patch_size
    flattened_patch_size = spec.num_channels * temporal_patch_size * patch_size * patch_size
    # Use 196 patches as template (14x14 grid from 224x224 image)
    num_patches = 196

    %{
      "pixel_values" => Nx.template({num_patches, flattened_patch_size}, :f32)
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
    # pixel_values from featurizer: {num_patches, channels * temporal * patch_h * patch_w}
    # This is the pre-extracted patch format like Python
    patch_size = spec.patch_size
    temporal_patch_size = spec.temporal_patch_size
    flattened_patch_size = spec.num_channels * temporal_patch_size * patch_size * patch_size
    pixel_shape = {nil, flattened_patch_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: pixel_shape)
    ])
  end

  defp core(inputs, spec) do
    pixel_values = inputs["pixel_values"]

    # Patch embedding: Apply Conv3d equivalent on pre-extracted patches
    # Python does: reshape {num_patches, 1536} -> {num_patches, C, T, H, W} -> Conv3d -> {num_patches, hidden_size}
    embeddings = patch_embedding(pixel_values, spec, name: "patch_embed")

    # Add learned position embeddings
    # Shape: {num_position_embeddings, hidden_size}
    embeddings = position_embedding(embeddings, spec, name: "pos_embed")

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

    # Input shape: {num_patches, channels * temporal_patch_size * patch_size * patch_size}
    # = {num_patches, 3 * 2 * 16 * 16} = {num_patches, 1536}
    #
    # Python PatchEmbed:
    # 1. Reshapes to {num_patches, C, T, H, W} = {num_patches, 3, 2, 16, 16}
    # 2. Applies Conv3d(3, 1024, kernel=(2,16,16), stride=(2,16,16))
    # 3. Output: {num_patches, 1024, 1, 1, 1} -> flatten to {num_patches, 1024}
    #
    # Since Conv3d with kernel=stride=full_size is equivalent to a linear projection,
    # we implement this as a dense layer.

    # Reshape for proper 3D conv simulation
    # {num_patches, 1536} -> {num_patches, 3, 2, 16, 16}
    reshaped =
      Axon.nx(pixel_values, fn x ->
        {num_patches, _flat} = Nx.shape(x)
        channels = spec.num_channels
        temporal = spec.temporal_patch_size
        patch_h = spec.patch_size
        patch_w = spec.patch_size
        Nx.reshape(x, {num_patches, channels, temporal, patch_h, patch_w})
      end)

    # Conv3d kernel param: {out_channels, in_channels, t, h, w}
    kernel_param =
      Axon.param(
        "kernel",
        fn _ ->
          {spec.hidden_size, spec.num_channels, spec.temporal_patch_size, spec.patch_size,
           spec.patch_size}
        end,
        initializer: kernel_initializer(spec)
      )

    # Conv3d bias param
    bias_param =
      Axon.param(
        "bias",
        fn _ -> {spec.hidden_size} end,
        initializer: Axon.Initializers.zeros()
      )

    # Apply Conv3d equivalent - since kernel covers entire input, it's like a dense layer
    Axon.layer(
      fn x, kernel, bias, _opts ->
        # x: {num_patches, 3, 2, 16, 16}
        # kernel: {hidden_size, 3, 2, 16, 16}
        # bias: {hidden_size}
        # Output: {num_patches, hidden_size}
        {num_patches, c, t, h, w} = Nx.shape(x)
        {hidden_size, _, _, _, _} = Nx.shape(kernel)

        # Flatten spatial dims: {num_patches, c*t*h*w}
        x_flat = Nx.reshape(x, {num_patches, c * t * h * w})
        # Flatten kernel: {hidden_size, c*t*h*w} -> transpose to {c*t*h*w, hidden_size}
        k_flat = Nx.reshape(kernel, {hidden_size, c * t * h * w})
        k_flat = Nx.transpose(k_flat)

        # Matrix multiply: {num_patches, c*t*h*w} @ {c*t*h*w, hidden_size} = {num_patches, hidden_size}
        result = Nx.dot(x_flat, k_flat)
        # Add bias
        Nx.add(result, bias)
      end,
      [reshaped, kernel_param, bias_param],
      name: join(name, "proj"),
      op_name: :conv3d
    )
    |> Axon.nx(fn x ->
      # Add batch dimension for transformer: {num_patches, hidden_size} -> {1, num_patches, hidden_size}
      Nx.new_axis(x, 0)
    end)
  end

  defp position_embedding(embeddings, spec, opts) do
    name = opts[:name]

    # Learned position embeddings: {num_position_embeddings, hidden_size}
    # num_position_embeddings = 2304 = 48*48 (a 2D grid of positions)
    # We need to interpolate to the actual grid size using bilinear interpolation
    pos_embed_param =
      Axon.param(
        "weight",
        fn _ -> {spec.num_position_embeddings, spec.hidden_size} end,
        initializer: kernel_initializer(spec)
      )

    Axon.layer(
      fn embed, pos_embed, _opts ->
        # embed: {batch, num_patches, hidden_size}
        # pos_embed: {num_position_embeddings, hidden_size} = {2304, 1024} = {48*48, 1024}
        {_batch, num_patches, _hidden_size} = Nx.shape(embed)

        # Compute target grid size (assuming square grid)
        grid_size = :math.sqrt(num_patches) |> trunc()

        # Source grid size (48x48)
        src_grid_size = :math.sqrt(spec.num_position_embeddings) |> trunc()

        # Bilinear interpolation from src_grid to target grid
        # For each patch at (row, col), compute interpolated position embedding

        # Create target grid indices
        h_idxs = Nx.linspace(0, src_grid_size - 1, n: grid_size, type: :f32)
        w_idxs = Nx.linspace(0, src_grid_size - 1, n: grid_size, type: :f32)

        # Floor and ceil indices
        h_floor = Nx.floor(h_idxs) |> Nx.as_type(:s32)
        w_floor = Nx.floor(w_idxs) |> Nx.as_type(:s32)
        h_ceil = Nx.add(h_floor, 1) |> Nx.min(src_grid_size - 1)
        w_ceil = Nx.add(w_floor, 1) |> Nx.min(src_grid_size - 1)

        # Interpolation weights
        dh = Nx.subtract(h_idxs, Nx.as_type(h_floor, :f32))
        dw = Nx.subtract(w_idxs, Nx.as_type(w_floor, :f32))

        # Compute indices into pos_embed (which is stored as 1D array of 48*48)
        # For a 2D grid position (r, c), the 1D index is r * src_grid_size + c

        # Create all (h, w) pairs for the target grid
        # We need indices for all 4 corners of each bilinear interpolation

        # Reshape for broadcasting: h indices along first dim, w along second
        h_floor_2d = Nx.reshape(h_floor, {grid_size, 1})
        h_ceil_2d = Nx.reshape(h_ceil, {grid_size, 1})
        w_floor_2d = Nx.reshape(w_floor, {1, grid_size})
        w_ceil_2d = Nx.reshape(w_ceil, {1, grid_size})

        # 4 corner indices (each is grid_size x grid_size)
        idx_ff = Nx.add(Nx.multiply(h_floor_2d, src_grid_size), w_floor_2d) |> Nx.flatten()
        idx_fc = Nx.add(Nx.multiply(h_floor_2d, src_grid_size), w_ceil_2d) |> Nx.flatten()
        idx_cf = Nx.add(Nx.multiply(h_ceil_2d, src_grid_size), w_floor_2d) |> Nx.flatten()
        idx_cc = Nx.add(Nx.multiply(h_ceil_2d, src_grid_size), w_ceil_2d) |> Nx.flatten()

        # Gather embeddings for all 4 corners
        emb_ff = Nx.take(pos_embed, idx_ff, axis: 0)
        emb_fc = Nx.take(pos_embed, idx_fc, axis: 0)
        emb_cf = Nx.take(pos_embed, idx_cf, axis: 0)
        emb_cc = Nx.take(pos_embed, idx_cc, axis: 0)

        # Compute bilinear weights (grid_size x grid_size -> flattened)
        dh_2d = Nx.reshape(dh, {grid_size, 1})
        dw_2d = Nx.reshape(dw, {1, grid_size})

        w_ff =
          Nx.multiply(Nx.subtract(1.0, dh_2d), Nx.subtract(1.0, dw_2d))
          |> Nx.flatten()
          |> Nx.reshape({num_patches, 1})

        w_fc =
          Nx.multiply(Nx.subtract(1.0, dh_2d), dw_2d)
          |> Nx.flatten()
          |> Nx.reshape({num_patches, 1})

        w_cf =
          Nx.multiply(dh_2d, Nx.subtract(1.0, dw_2d))
          |> Nx.flatten()
          |> Nx.reshape({num_patches, 1})

        w_cc = Nx.multiply(dh_2d, dw_2d) |> Nx.flatten() |> Nx.reshape({num_patches, 1})

        # Weighted sum for interpolated embeddings
        interpolated =
          Nx.add(
            Nx.add(
              Nx.multiply(emb_ff, w_ff),
              Nx.multiply(emb_fc, w_fc)
            ),
            Nx.add(
              Nx.multiply(emb_cf, w_cf),
              Nx.multiply(emb_cc, w_cc)
            )
          )

        # Add to embeddings (broadcast to batch dimension)
        Nx.add(embed, interpolated)
      end,
      [embeddings, pos_embed_param],
      name: name,
      op_name: :position_embedding
    )
  end

  defp encoder(embeddings, spec, opts) do
    name = opts[:name]

    # Convert deepstack indexes to 0-indexed
    deepstack_indexes =
      spec.deepstack_visual_indexes
      |> Enum.map(&(&1 - 1))
      |> MapSet.new()

    # Qwen3-VL uses 2D spatial rotary embeddings where each patch has (row, col) position.
    # Python's rot_pos_emb computes row and col frequencies separately, then concatenates them.
    #
    # For each patch at position (row, col):
    # - First half of rotary_dim: row_position * inv_freq
    # - Second half of rotary_dim: col_position * inv_freq
    #
    # We compute 2D rotary embeddings (cos, sin) for all patches based on their grid position.
    rotary_2d =
      Axon.nx(embeddings, fn embed ->
        {_batch, seq_len, _hidden} = Nx.shape(embed)
        grid_size = :math.sqrt(seq_len) |> trunc()
        head_dim = div(spec.hidden_size, spec.num_attention_heads)
        rotary_dim = div(head_dim, 2)

        compute_2d_rotary_embedding(seq_len, grid_size, rotary_dim, spec.rotary_embedding_base)
      end)

    # Use custom transformer blocks with 2D rotary embedding
    # Since Layers.Transformer.blocks only supports 1D position-based rotary,
    # we implement vision transformer blocks directly
    vision_transformer_blocks(embeddings, rotary_2d, spec, deepstack_indexes, name)
  end

  # Compute 2D rotary embedding (cos, sin) for vision patches
  # Returns {cos, sin} each of shape {seq_len, rotary_dim}
  defnp compute_2d_rotary_embedding(seq_len, grid_size, rotary_dim, base) do
    # For each patch in raster scan order, compute (row, col) position
    positions = Nx.iota({seq_len})
    row_positions = Nx.quotient(positions, grid_size)
    col_positions = Nx.remainder(positions, grid_size)

    # Compute inverse frequencies (half rotary_dim because we split for row/col)
    half_rotary_dim = div(rotary_dim, 2)
    range = Nx.iota({half_rotary_dim}) |> Nx.multiply(2) |> Nx.divide(rotary_dim)
    inv_freq = 1.0 / Nx.pow(base, range)

    # Compute angles for rows and columns
    # row_angles: {seq_len, half_rotary_dim}
    row_angles = Nx.outer(row_positions, inv_freq)
    col_angles = Nx.outer(col_positions, inv_freq)

    # Concatenate row and col angles: {seq_len, rotary_dim}
    angles = Nx.concatenate([row_angles, col_angles], axis: -1)

    # Compute cos and sin
    cos = Nx.cos(angles)
    sin = Nx.sin(angles)

    {cos, sin}
  end

  # Custom vision transformer blocks with 2D rotary embedding
  defp vision_transformer_blocks(embeddings, rotary_2d, spec, deepstack_indexes, name) do
    head_dim = div(spec.hidden_size, spec.num_attention_heads)

    # Build blocks iteratively, collecting hidden states for deepstack
    {hidden_state, hidden_states, attentions} =
      Enum.reduce(0..(spec.num_blocks - 1), {embeddings, [], []}, fn idx,
                                                                     {hidden_state, hidden_states,
                                                                      attentions} ->
        block_name = join(name, idx)

        # Pre-norm
        normed =
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(block_name, "norm1")
          )

        # Self-attention with 2D rotary
        {attn_output, attn_weights} =
          vision_attention_with_2d_rotary(
            normed,
            rotary_2d,
            spec,
            head_dim,
            join(block_name, "attn")
          )

        hidden_state = Axon.add(hidden_state, attn_output)

        # FFN with pre-norm
        normed =
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(block_name, "norm2")
          )

        ffn_output =
          normed
          |> Axon.dense(spec.intermediate_size,
            kernel_initializer: kernel_initializer(spec),
            name: join(block_name, "mlp.fc1")
          )
          |> Layers.activation(spec.activation)
          |> Axon.dense(spec.hidden_size,
            kernel_initializer: kernel_initializer(spec),
            name: join(block_name, "mlp.fc2")
          )

        hidden_state = Axon.add(hidden_state, ffn_output)

        hidden_states = hidden_states ++ [hidden_state]
        attentions = attentions ++ [attn_weights]

        {hidden_state, hidden_states, attentions}
      end)

    # Extract deepstack hidden states
    deepstack_hidden_states =
      deepstack_indexes
      |> Enum.sort()
      |> Enum.map(fn idx ->
        if idx < length(hidden_states) do
          Enum.at(hidden_states, idx)
        else
          List.last(hidden_states)
        end
      end)

    %{
      hidden_state: hidden_state,
      hidden_states: Axon.container(List.to_tuple(hidden_states)),
      attentions: Axon.container(List.to_tuple(attentions)),
      deepstack_hidden_states: Axon.container(List.to_tuple(deepstack_hidden_states))
    }
  end

  # Vision attention with 2D rotary embedding
  defp vision_attention_with_2d_rotary(hidden_state, rotary_2d, spec, head_dim, name) do
    # QKV projection (combined)
    qkv =
      Axon.dense(hidden_state, spec.hidden_size * 3,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "qkv")
      )

    # Split and reshape for multi-head attention
    {query, key, value} =
      Axon.layer(
        fn qkv, _opts ->
          {batch, seq_len, _} = Nx.shape(qkv)
          qkv_reshaped = Nx.reshape(qkv, {batch, seq_len, 3, spec.num_attention_heads, head_dim})
          qkv_transposed = Nx.transpose(qkv_reshaped, axes: [2, 0, 3, 1, 4])
          # {3, batch, heads, seq, head_dim}
          {qkv_transposed[0], qkv_transposed[1], qkv_transposed[2]}
        end,
        [qkv],
        name: join(name, "split_qkv")
      )
      |> then(fn layer ->
        q = Axon.nx(layer, fn {q, _k, _v} -> q end)
        k = Axon.nx(layer, fn {_q, k, _v} -> k end)
        v = Axon.nx(layer, fn {_q, _k, v} -> v end)
        {q, k, v}
      end)

    # Apply 2D rotary embedding to query and key
    {rotated_query, rotated_key} =
      Axon.layer(
        fn query, key, rotary_2d, _opts ->
          {cos, sin} = rotary_2d
          apply_2d_rotary_embedding(query, key, cos, sin)
        end,
        [query, key, rotary_2d],
        name: join(name, "rotary_2d")
      )
      |> then(fn layer ->
        q = Axon.nx(layer, fn {q, _k} -> q end)
        k = Axon.nx(layer, fn {_q, k} -> k end)
        {q, k}
      end)

    # Scaled dot-product attention
    scale = :math.sqrt(head_dim)

    attn_output =
      Axon.layer(
        fn query, key, value, _opts ->
          # query, key, value: {batch, heads, seq, head_dim}
          # Attention scores: {batch, heads, seq, seq}
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)
          weights = Axon.Activations.softmax(scores, axis: -1)

          # Weighted sum: {batch, heads, seq, head_dim}
          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          {output, weights}
        end,
        [rotated_query, rotated_key, value],
        name: join(name, "attention")
      )

    output = Axon.nx(attn_output, fn {out, _weights} -> out end)
    weights = Axon.nx(attn_output, fn {_out, weights} -> weights end)

    # Reshape and project output
    output =
      Axon.layer(
        fn x, _opts ->
          {batch, heads, seq_len, head_dim} = Nx.shape(x)
          hidden_size = heads * head_dim

          x
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch, seq_len, hidden_size})
        end,
        [output],
        name: join(name, "reshape_output")
      )

    output =
      Axon.dense(output, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "proj")
      )

    {output, weights}
  end

  # Apply 2D rotary embedding to query and key
  # cos, sin: {seq_len, rotary_dim}
  # query, key: {batch, heads, seq_len, head_dim}
  defnp apply_2d_rotary_embedding(query, key, cos, sin) do
    # Rotary embedding only applies to first half of head_dim
    {_batch, _heads, _seq, head_dim} = Nx.shape(query)
    rotary_dim = div(head_dim, 2)

    # Split query/key into rotary and non-rotary parts
    {q_rot, q_pass} = split_rotary(query, rotary_dim)
    {k_rot, k_pass} = split_rotary(key, rotary_dim)

    # Expand cos/sin for broadcasting: {1, 1, seq_len, rotary_dim}
    cos = cos |> Nx.new_axis(0) |> Nx.new_axis(0)
    sin = sin |> Nx.new_axis(0) |> Nx.new_axis(0)

    # Apply rotary embedding
    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin

    # Concatenate back
    rotated_q = Nx.concatenate([q_embed, q_pass], axis: -1)
    rotated_k = Nx.concatenate([k_embed, k_pass], axis: -1)

    {rotated_q, rotated_k}
  end

  defnp split_rotary(tensor, rotary_dim) do
    {batch, heads, seq, head_dim} = Nx.shape(tensor)
    pass_dim = head_dim - rotary_dim
    rotary_part = Nx.slice(tensor, [0, 0, 0, 0], [batch, heads, seq, rotary_dim])
    pass_part = Nx.slice(tensor, [0, 0, 0, rotary_dim], [batch, heads, seq, pass_dim])
    {rotary_part, pass_part}
  end

  defnp rotate_half(x) do
    # Split in half along last dimension and swap with negation
    {batch, heads, seq, dim} = Nx.shape(x)
    half_dim = div(dim, 2)
    x1 = Nx.slice(x, [0, 0, 0, 0], [batch, heads, seq, half_dim])
    x2 = Nx.slice(x, [0, 0, 0, half_dim], [batch, heads, seq, half_dim])
    Nx.concatenate([Nx.negate(x2), x1], axis: -1)
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

      # Vision config uses embed_dim (Qwen2-VL) or hidden_size (Qwen3-VL)
      opts =
        convert!(data,
          num_blocks: {"depth", number()},
          num_attention_heads: {"num_heads", number()},
          num_channels: {"in_channels", number()},
          patch_size: {"patch_size", number()},
          temporal_patch_size: {"temporal_patch_size", number()},
          spatial_merge_size: {"spatial_merge_size", number()},
          activation: {"hidden_act", activation()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      # Handle both embed_dim (Qwen2-VL) and hidden_size (Qwen3-VL)
      hidden_size = data["hidden_size"] || data["embed_dim"] || spec.hidden_size
      opts = Keyword.put(opts, :hidden_size, hidden_size)

      # Compute derived values
      # intermediate_size from config or computed as hidden_size * mlp_ratio (default mlp_ratio = 4)
      mlp_ratio = Map.get(data, "mlp_ratio", 4)
      intermediate_size = data["intermediate_size"] || hidden_size * mlp_ratio

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
        # Patch embedding - keep 3D conv kernel as-is
        # PyTorch Conv3d weight shape: {out_channels, in_channels, temporal, h, w} = {1024, 3, 2, 16, 16}
        # Our custom layer expects the same shape
        "patch_embed.proj" => %{
          "kernel" => {
            [{"visual.patch_embed.proj", "weight"}],
            fn [kernel] ->
              # Keep in PyTorch format: {out_channels, in_channels, t, h, w}
              kernel
            end
          },
          "bias" => {
            [{"visual.patch_embed.proj", "bias"}],
            fn [bias] -> bias end
          }
        },
        # Learned position embeddings
        "pos_embed" => "visual.pos_embed",
        # Transformer blocks - using custom 2D rotary attention
        "blocks.{n}.norm1" => "visual.blocks.{n}.norm1",
        "blocks.{n}.attn.qkv" => "visual.blocks.{n}.attn.qkv",
        "blocks.{n}.attn.proj" => "visual.blocks.{n}.attn.proj",
        "blocks.{n}.norm2" => "visual.blocks.{n}.norm2",
        "blocks.{n}.mlp.fc1" => "visual.blocks.{n}.mlp.linear_fc1",
        "blocks.{n}.mlp.fc2" => "visual.blocks.{n}.mlp.linear_fc2",
        # Patch merger - Qwen3VL uses linear_fc1/fc2/norm naming
        "merger.ln_q" => "visual.merger.norm",
        "merger.mlp.0" => "visual.merger.linear_fc1",
        "merger.mlp.2" => "visual.merger.linear_fc2"
      }
    end
  end
end
