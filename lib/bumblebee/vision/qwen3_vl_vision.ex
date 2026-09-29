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
        doc: "the number of learned absolute position embeddings (a square grid)"
      ],
      deepstack_visual_indexes: [
        default: [5, 11, 17],
        doc:
          "the encoder layer indices from which to extract DeepStack features (0-indexed, matching HuggingFace's `enumerate(self.blocks)`)"
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

  Patches arrive from the featurizer in windowed order: every group of
  `spatial_merge_size ** 2` consecutive patches forms a contiguous spatial
  merge block. Combined with the per-image `image_grid_thw` tensor, this
  encoder supports a variable number of images of varying sizes in a
  single forward pass.

  ## Architectures

    * `:base` - the base vision encoder model

  ## Inputs

    * `"pixel_values"` - `{num_patches, num_channels * temporal_patch_size * patch_size * patch_size}`

      Concatenated, pre-extracted image/video patches from the featurizer.

    * `"image_grid_thw"` - `{num_images, 3}`

      Per-image grid dimensions `[temporal, height, width]` in patch
      units, used to derive per-patch row/column positions for the
      learned bilinear position embedding and the 2D rotary embedding.

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
    patch_size = spec.patch_size
    temporal_patch_size = spec.temporal_patch_size
    flattened_patch_size = spec.num_channels * temporal_patch_size * patch_size * patch_size
    # 14x14 grid from a 224x224 image with patch_size=16
    num_patches = 196

    %{
      "pixel_values" => Nx.template({num_patches, flattened_patch_size}, :f32),
      "image_grid_thw" => Nx.template({1, 3}, :s64)
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
    patch_size = spec.patch_size
    temporal_patch_size = spec.temporal_patch_size
    flattened_patch_size = spec.num_channels * temporal_patch_size * patch_size * patch_size

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: {nil, flattened_patch_size}),
      Axon.input("image_grid_thw", shape: {nil, 3})
    ])
  end

  defp core(inputs, spec) do
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    embeddings =
      pixel_values
      |> patch_embedding(spec, name: "patch_embed")
      |> position_embedding(grid_thw, spec, name: "pos_embed")

    encoder_outputs = encoder(embeddings, grid_thw, spec, name: "blocks")

    hidden_state = patch_merger(encoder_outputs.hidden_state, spec, name: "merger")

    %{
      hidden_state: hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      deepstack_hidden_states: encoder_outputs.deepstack_hidden_states
    }
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    # Input: {num_patches, channels * temporal_patch_size * patch_size * patch_size}
    # PyTorch's Conv3d with kernel=stride=full_patch is equivalent to a dense projection
    # over the flattened patch features. The kernel param keeps PyTorch's
    # {out_channels, in_channels, t, h, w} layout for clean weight loading.
    reshaped =
      Axon.nx(pixel_values, fn x ->
        {num_patches, _flat} = Nx.shape(x)

        Nx.reshape(
          x,
          {num_patches, spec.num_channels, spec.temporal_patch_size, spec.patch_size,
           spec.patch_size}
        )
      end)

    kernel_param =
      Axon.param(
        "kernel",
        fn _ ->
          {spec.hidden_size, spec.num_channels, spec.temporal_patch_size, spec.patch_size,
           spec.patch_size}
        end,
        initializer: kernel_initializer(spec)
      )

    bias_param =
      Axon.param("bias", fn _ -> {spec.hidden_size} end, initializer: Axon.Initializers.zeros())

    Axon.layer(
      fn x, kernel, bias, _opts ->
        {num_patches, c, t, h, w} = Nx.shape(x)
        {hidden_size, _, _, _, _} = Nx.shape(kernel)

        x_flat = Nx.reshape(x, {num_patches, c * t * h * w})
        k_flat = kernel |> Nx.reshape({hidden_size, c * t * h * w}) |> Nx.transpose()

        x_flat
        |> Nx.dot(k_flat)
        |> Nx.add(bias)
      end,
      [reshaped, kernel_param, bias_param],
      name: join(name, "proj"),
      op_name: :conv3d
    )
    |> Axon.nx(fn x -> Nx.new_axis(x, 0) end)
  end

  defp position_embedding(embeddings, grid_thw, spec, opts) do
    name = opts[:name]

    pos_embed_param =
      Axon.param(
        "weight",
        fn _, _ -> {spec.num_position_embeddings, spec.hidden_size} end,
        initializer: kernel_initializer(spec)
      )

    Axon.layer(
      fn embed, grid_thw_t, pos_embed, _opts ->
        bilinear_interpolated_position(embed, grid_thw_t, pos_embed, spec)
      end,
      [embeddings, grid_thw, pos_embed_param],
      name: name,
      op_name: :position_embedding
    )
  end

  defp bilinear_interpolated_position(embed, grid_thw, pos_embed, spec) do
    {_batch, total_patches, _hidden} = Nx.shape(embed)
    src_grid_size = trunc(:math.sqrt(spec.num_position_embeddings))
    merge_size = spec.spatial_merge_size

    {row_in_image, col_in_image, grid_h_per_patch, grid_w_per_patch, _image_id, _patch_valid} =
      patch_metadata(grid_thw, total_patches, merge_size)

    src_max_f = Nx.tensor(src_grid_size - 1, type: :f32)

    grid_h_minus_one = grid_h_per_patch |> Nx.subtract(1) |> Nx.max(1) |> Nx.as_type(:f32)
    grid_w_minus_one = grid_w_per_patch |> Nx.subtract(1) |> Nx.max(1) |> Nx.as_type(:f32)

    row_src_f =
      row_in_image
      |> Nx.as_type(:f32)
      |> Nx.multiply(src_max_f)
      |> Nx.divide(grid_h_minus_one)

    col_src_f =
      col_in_image
      |> Nx.as_type(:f32)
      |> Nx.multiply(src_max_f)
      |> Nx.divide(grid_w_minus_one)

    row_src_f = Nx.select(Nx.equal(grid_h_per_patch, 1), Nx.tensor(0.0), row_src_f)
    col_src_f = Nx.select(Nx.equal(grid_w_per_patch, 1), Nx.tensor(0.0), col_src_f)

    row_floor = row_src_f |> Nx.floor() |> Nx.as_type(:s32)
    col_floor = col_src_f |> Nx.floor() |> Nx.as_type(:s32)
    row_ceil = row_floor |> Nx.add(1) |> Nx.min(src_grid_size - 1)
    col_ceil = col_floor |> Nx.add(1) |> Nx.min(src_grid_size - 1)

    dh = Nx.subtract(row_src_f, Nx.as_type(row_floor, :f32))
    dw = Nx.subtract(col_src_f, Nx.as_type(col_floor, :f32))

    idx_ff = row_floor |> Nx.multiply(src_grid_size) |> Nx.add(col_floor)
    idx_fc = row_floor |> Nx.multiply(src_grid_size) |> Nx.add(col_ceil)
    idx_cf = row_ceil |> Nx.multiply(src_grid_size) |> Nx.add(col_floor)
    idx_cc = row_ceil |> Nx.multiply(src_grid_size) |> Nx.add(col_ceil)

    emb_ff = Nx.take(pos_embed, idx_ff, axis: 0)
    emb_fc = Nx.take(pos_embed, idx_fc, axis: 0)
    emb_cf = Nx.take(pos_embed, idx_cf, axis: 0)
    emb_cc = Nx.take(pos_embed, idx_cc, axis: 0)

    w_ff = dh |> Nx.subtract(1.0) |> Nx.negate() |> Nx.multiply(Nx.subtract(1.0, dw))
    w_fc = dh |> Nx.subtract(1.0) |> Nx.negate() |> Nx.multiply(dw)
    w_cf = Nx.multiply(dh, Nx.subtract(1.0, dw))
    w_cc = Nx.multiply(dh, dw)

    interpolated =
      Nx.multiply(emb_ff, Nx.new_axis(w_ff, -1))
      |> Nx.add(Nx.multiply(emb_fc, Nx.new_axis(w_fc, -1)))
      |> Nx.add(Nx.multiply(emb_cf, Nx.new_axis(w_cf, -1)))
      |> Nx.add(Nx.multiply(emb_cc, Nx.new_axis(w_cc, -1)))

    Nx.add(embed, interpolated)
  end

  # Per-patch metadata derived from image_grid_thw.
  # Returns {row_in_image, col_in_image, grid_h_per_patch, grid_w_per_patch, image_id_per_patch}.
  # All tensors have shape {total_patches}.
  defp patch_metadata(grid_thw, total_patches, merge_size) do
    grid_t = grid_thw[[.., 0]]
    grid_h = grid_thw[[.., 1]]
    grid_w = grid_thw[[.., 2]]

    patches_per_image = grid_t |> Nx.multiply(grid_h) |> Nx.multiply(grid_w)

    cumulative = Nx.cumulative_sum(patches_per_image)
    exclusive_cumulative = Nx.subtract(cumulative, patches_per_image)
    total_real_patches = Nx.sum(patches_per_image)

    patch_indices = Nx.iota({total_patches}, type: :s64)

    # Patches beyond total_real_patches are padding slots (when the
    # featurizer was configured with :max_patches). Mark them invalid so
    # downstream attention masking can exclude them entirely.
    patch_valid = Nx.less(patch_indices, total_real_patches)

    image_id_raw =
      patch_indices
      |> Nx.new_axis(-1)
      |> Nx.greater_equal(Nx.new_axis(cumulative, 0))
      |> Nx.sum(axes: [-1])
      |> Nx.as_type(:s64)

    n_images = Nx.axis_size(grid_thw, 0)
    # Padded patches map to image_id == n_images (out of bounds). Clip so
    # gather operations succeed. Their derived row/col/grid values are
    # garbage but get masked out via `patch_valid` in the attention step.
    image_id_per_patch = Nx.clip(image_id_raw, 0, n_images - 1)

    offset_per_patch = Nx.take(exclusive_cumulative, image_id_per_patch)
    local_index = Nx.subtract(patch_indices, offset_per_patch)

    grid_h_per_patch = Nx.take(grid_h, image_id_per_patch)
    grid_w_per_patch = Nx.take(grid_w, image_id_per_patch)

    # Padded images have grid_w == 0; guard the divisions so we don't
    # divide by zero. The resulting coordinates for padded patches are
    # arbitrary and are masked out downstream.
    safe_grid_w = Nx.max(grid_w_per_patch, merge_size)

    merge_sq = merge_size * merge_size
    merged_w_per_patch = Nx.quotient(safe_grid_w, merge_size)

    block_idx = Nx.quotient(local_index, merge_sq)
    within = Nx.remainder(local_index, merge_sq)
    block_row = Nx.quotient(block_idx, merged_w_per_patch)
    block_col = Nx.remainder(block_idx, merged_w_per_patch)
    within_h = Nx.quotient(within, merge_size)
    within_w = Nx.remainder(within, merge_size)

    row_in_image = block_row |> Nx.multiply(merge_size) |> Nx.add(within_h)
    col_in_image = block_col |> Nx.multiply(merge_size) |> Nx.add(within_w)

    {row_in_image, col_in_image, grid_h_per_patch, grid_w_per_patch, image_id_per_patch,
     patch_valid}
  end

  defp encoder(embeddings, grid_thw, spec, opts) do
    name = opts[:name]

    deepstack_indexes = MapSet.new(spec.deepstack_visual_indexes)

    head_dim = div(spec.hidden_size, spec.num_attention_heads)
    rotary_dim = div(head_dim, 2)

    rotary_2d =
      Axon.layer(
        fn embed, grid_thw_t, _opts ->
          {_batch, total_patches, _hidden} = Nx.shape(embed)

          {row_in_image, col_in_image, _, _, _, _} =
            patch_metadata(grid_thw_t, total_patches, spec.spatial_merge_size)

          compute_2d_rotary_from_positions(
            row_in_image,
            col_in_image,
            rotary_dim,
            spec.rotary_embedding_base
          )
        end,
        [embeddings, grid_thw],
        op_name: :rotary_2d
      )

    attention_mask =
      Axon.layer(
        fn embed, grid_thw_t, _opts ->
          {_batch, total_patches, _hidden} = Nx.shape(embed)

          {_, _, _, _, image_id_per_patch, patch_valid} =
            patch_metadata(grid_thw_t, total_patches, spec.spatial_merge_size)

          block_diagonal_attention_mask(image_id_per_patch, patch_valid)
        end,
        [embeddings, grid_thw],
        op_name: :attention_mask
      )

    vision_transformer_blocks(
      embeddings,
      rotary_2d,
      attention_mask,
      spec,
      deepstack_indexes,
      name
    )
  end

  # 2D rotary cos/sin from per-patch (row, col) positions.
  # Returns {cos, sin}, each of shape {total_patches, rotary_dim}.
  defnp compute_2d_rotary_from_positions(row_positions, col_positions, rotary_dim, base) do
    half_rotary_dim = div(rotary_dim, 2)
    range = Nx.iota({half_rotary_dim}) |> Nx.multiply(2) |> Nx.divide(rotary_dim)
    inv_freq = 1.0 / Nx.pow(base, range)

    row_angles = Nx.outer(Nx.as_type(row_positions, :f32), inv_freq)
    col_angles = Nx.outer(Nx.as_type(col_positions, :f32), inv_freq)

    angles = Nx.concatenate([row_angles, col_angles], axis: -1)
    {Nx.cos(angles), Nx.sin(angles)}
  end

  # Returns {total_patches, total_patches} boolean tensor where True means
  # the two patches share an image AND both are valid (not padding).
  defnp block_diagonal_attention_mask(image_id_per_patch, patch_valid) do
    a = Nx.new_axis(image_id_per_patch, -1)
    b = Nx.new_axis(image_id_per_patch, 0)
    same_image = Nx.equal(a, b)
    valid_pair = Nx.multiply(Nx.new_axis(patch_valid, -1), Nx.new_axis(patch_valid, 0))
    Nx.logical_and(same_image, valid_pair)
  end

  defp vision_transformer_blocks(
         embeddings,
         rotary_2d,
         attention_mask,
         spec,
         deepstack_indexes,
         name
       ) do
    head_dim = div(spec.hidden_size, spec.num_attention_heads)

    {hidden_state, hidden_states, attentions} =
      Enum.reduce(0..(spec.num_blocks - 1), {embeddings, [], []}, fn idx,
                                                                     {hidden_state, hidden_states,
                                                                      attentions} ->
        block_name = join(name, idx)

        normed =
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(block_name, "norm1")
          )

        {attn_output, attn_weights} =
          vision_attention_with_2d_rotary(
            normed,
            rotary_2d,
            attention_mask,
            spec,
            head_dim,
            join(block_name, "attn")
          )

        hidden_state = Axon.add(hidden_state, attn_output)

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

        {hidden_state, hidden_states ++ [hidden_state], attentions ++ [attn_weights]}
      end)

    deepstack_merged_features =
      deepstack_indexes
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.map(fn {layer_idx, merger_idx} ->
        hidden_state_at_layer =
          if layer_idx < length(hidden_states) do
            Enum.at(hidden_states, layer_idx)
          else
            List.last(hidden_states)
          end

        deepstack_merger(hidden_state_at_layer, spec, merger_idx, "deepstack_merger_list")
      end)

    %{
      hidden_state: hidden_state,
      hidden_states: Axon.container(List.to_tuple(hidden_states)),
      attentions: Axon.container(List.to_tuple(attentions)),
      deepstack_hidden_states: Axon.container(List.to_tuple(deepstack_merged_features))
    }
  end

  defp deepstack_merger(hidden_state, spec, index, name) do
    merger_name = join(name, index)
    merge_sq = spec.spatial_merge_size * spec.spatial_merge_size
    mlp_input_size = spec.hidden_size * merge_sq

    hidden_state
    |> Axon.nx(fn x ->
      {batch, total_patches, hidden} = Nx.shape(x)
      Nx.reshape(x, {batch, div(total_patches, merge_sq), merge_sq * hidden})
    end)
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(merger_name, "norm"))
    |> Axon.dense(mlp_input_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(merger_name, "linear_fc1")
    )
    |> Layers.activation(spec.activation)
    |> Axon.dense(spec.out_hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(merger_name, "linear_fc2")
    )
  end

  defp vision_attention_with_2d_rotary(
         hidden_state,
         rotary_2d,
         attention_mask,
         spec,
         head_dim,
         name
       ) do
    qkv =
      Axon.dense(hidden_state, spec.hidden_size * 3,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "qkv")
      )

    {query, key, value} =
      Axon.layer(
        fn qkv, _opts ->
          {batch, seq_len, _} = Nx.shape(qkv)
          qkv_reshaped = Nx.reshape(qkv, {batch, seq_len, 3, spec.num_attention_heads, head_dim})
          qkv_transposed = Nx.transpose(qkv_reshaped, axes: [2, 0, 3, 1, 4])
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

    scale = :math.sqrt(head_dim)

    attn_output =
      Axon.layer(
        fn query, key, value, attention_mask, _opts ->
          # query, key, value: {batch, heads, seq, head_dim}
          # attention_mask: {seq, seq} boolean (True = attend)
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)

          mask_value =
            attention_mask
            |> Nx.select(Nx.tensor(0.0, type: :f32), Nx.tensor(-1.0e9, type: :f32))
            |> Nx.new_axis(0)
            |> Nx.new_axis(0)

          scores = Nx.add(scores, mask_value)
          weights = Axon.Activations.softmax(scores, axis: -1)
          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          {output, weights}
        end,
        [rotated_query, rotated_key, value, attention_mask],
        name: join(name, "attention")
      )

    output = Axon.nx(attn_output, fn {out, _weights} -> out end)
    weights = Axon.nx(attn_output, fn {_out, weights} -> weights end)

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

  defnp apply_2d_rotary_embedding(query, key, cos, sin) do
    {_batch, _heads, _seq, head_dim} = Nx.shape(query)
    rotary_dim = div(head_dim, 2)

    {q_rot, q_pass} = split_rotary(query, rotary_dim)
    {k_rot, k_pass} = split_rotary(key, rotary_dim)

    cos = cos |> Nx.new_axis(0) |> Nx.new_axis(0)
    sin = sin |> Nx.new_axis(0) |> Nx.new_axis(0)

    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin

    {Nx.concatenate([q_embed, q_pass], axis: -1), Nx.concatenate([k_embed, k_pass], axis: -1)}
  end

  defnp split_rotary(tensor, rotary_dim) do
    {batch, heads, seq, head_dim} = Nx.shape(tensor)
    pass_dim = head_dim - rotary_dim
    rotary_part = Nx.slice(tensor, [0, 0, 0, 0], [batch, heads, seq, rotary_dim])
    pass_part = Nx.slice(tensor, [0, 0, 0, rotary_dim], [batch, heads, seq, pass_dim])
    {rotary_part, pass_part}
  end

  defnp rotate_half(x) do
    {batch, heads, seq, dim} = Nx.shape(x)
    half_dim = div(dim, 2)
    x1 = Nx.slice(x, [0, 0, 0, 0], [batch, heads, seq, half_dim])
    x2 = Nx.slice(x, [0, 0, 0, half_dim], [batch, heads, seq, half_dim])
    Nx.concatenate([Nx.negate(x2), x1], axis: -1)
  end

  defp patch_merger(hidden_state, spec, opts) do
    name = opts[:name]
    merge_sq = spec.spatial_merge_size * spec.spatial_merge_size
    mlp_input_size = spec.hidden_size * merge_sq

    hidden_state
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "ln_q"))
    |> Axon.nx(fn x ->
      {batch, total_patches, hidden} = Nx.shape(x)
      Nx.reshape(x, {batch, div(total_patches, merge_sq), merge_sq * hidden})
    end)
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
    def load(spec, %{"model_type" => "qwen3_vl", "vision_config" => data}) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

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

      hidden_size = data["hidden_size"] || data["embed_dim"] || spec.hidden_size
      opts = Keyword.put(opts, :hidden_size, hidden_size)

      mlp_ratio = Map.get(data, "mlp_ratio", 4)
      intermediate_size = data["intermediate_size"] || hidden_size * mlp_ratio
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
        "patch_embed.proj" => %{
          "kernel" => {
            [{"visual.patch_embed.proj", "weight"}],
            fn [kernel] -> kernel end
          },
          "bias" => {
            [{"visual.patch_embed.proj", "bias"}],
            fn [bias] -> bias end
          }
        },
        "pos_embed" => "visual.pos_embed",
        "blocks.{n}.norm1" => "visual.blocks.{n}.norm1",
        "blocks.{n}.attn.qkv" => "visual.blocks.{n}.attn.qkv",
        "blocks.{n}.attn.proj" => "visual.blocks.{n}.attn.proj",
        "blocks.{n}.norm2" => "visual.blocks.{n}.norm2",
        "blocks.{n}.mlp.fc1" => "visual.blocks.{n}.mlp.linear_fc1",
        "blocks.{n}.mlp.fc2" => "visual.blocks.{n}.mlp.linear_fc2",
        "merger.ln_q" => "visual.merger.norm",
        "merger.mlp.0" => "visual.merger.linear_fc1",
        "merger.mlp.2" => "visual.merger.linear_fc2",
        "deepstack_merger_list.{n}.norm" => "visual.deepstack_merger_list.{n}.norm",
        "deepstack_merger_list.{n}.linear_fc1" => "visual.deepstack_merger_list.{n}.linear_fc1",
        "deepstack_merger_list.{n}.linear_fc2" => "visual.deepstack_merger_list.{n}.linear_fc2"
      }
    end
  end
end
