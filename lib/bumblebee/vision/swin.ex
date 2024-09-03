defmodule Bumblebee.Vision.Swin do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 224,
        doc: "the size of the input spatial dimensions"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 4,
        doc: "the size of the patch spatial dimensions"
      ],
      embedding_size: [
        default: 96,
        doc: "the dimensionality of patch embedding layer"
      ],
      use_absolute_position_embeddings: [
        default: false,
        doc: "whether to add absolute position embeddings to the patch embeddings"
      ],
      num_blocks: [
        default: [2, 2, 6, 2],
        doc: "the number of Transformer blocks in the encoder at each stage"
      ],
      num_attention_heads: [
        default: [3, 6, 12, 24],
        doc: "the number of attention heads for each attention layer in the encoder at each stage"
      ],
      window_size: [
        default: 7,
        doc:
          "the window size, used to limit self-attention computation to non-overlapping windows"
      ],
      intermediate_size_ratio: [
        default: 4,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder,
        expressed as a multiplier of hidden size (at the given stage)
        """
      ],
      use_attention_bias: [
        default: true,
        doc: "whether to use bias in query, key, and value projections"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      drop_path_rate: [
        default: 0.1,
        doc: "the drop path rate used to for stochastic depth"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  Swin Transformer model.

  ## Architectures

    * `:base` - plain Swin without any head on top

    * `:for_image_classification` - Swin tranformer model with a
      classification head

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]
  import Nx.Defn

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_image_classification]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(spec) do
    %{
      "pixel_values" =>
        Nx.template({1, spec.image_size, spec.image_size, spec.num_channels}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    spec
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.pooled_state, spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "image_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape),
      Axon.input("patch_mask", shape: {nil, nil}, optional: true)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["pixel_values"], inputs["patch_mask"], spec, name: join(name, "embedder"))

    encoder_outputs =
      encoder(embeddings, spec, name: join(name, "encoder"))

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "norm")
      )

    pooled_state =
      hidden_state
      |> Axon.adaptive_avg_pool(output_size: {1}, name: join(name, "pooler"))
      |> Axon.flatten()

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(pixel_values, patch_mask, spec, opts) do
    name = opts[:name]

    embeddings =
      pixel_values
      |> patch_embedding(spec, name: join(name, "patch_embedding"))
      |> Layers.apply_vision_patch_mask(patch_mask, name: join(name, "mask_tokens"))
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))

    embeddings =
      if spec.use_absolute_position_embeddings do
        num_patches = div(spec.image_size, spec.patch_size) ** 2

        position_embeddings =
          Layers.learned_embeddings(num_patches, spec.embedding_size,
            initializer: :zeros,
            name: join(name, "position_embedding")
          )

        Axon.add(embeddings, position_embeddings)
      else
        embeddings
      end

    embeddings
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]
    hidden_size = spec.embedding_size

    pixel_values
    |> Axon.conv(hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection")
    )
    |> Axon.reshape({:batch, :auto, spec.embedding_size}, name: join(name, "reshape"))
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({})
    }

    for stage_idx <- 0..(length(spec.num_blocks) - 1), reduce: state do
      state ->
        name = name |> join("stages") |> join(stage_idx)

        grid_size = div(spec.image_size, spec.patch_size)
        input_resolution = div(grid_size, 2 ** stage_idx)

        {hidden_state, attention, hidden_state_before_downsample} =
          stage(state.hidden_state, spec,
            hidden_size: spec.embedding_size * 2 ** stage_idx,
            num_blocks: Enum.at(spec.num_blocks, stage_idx),
            num_attention_heads: Enum.at(spec.num_attention_heads, stage_idx),
            downsample: stage_idx < length(spec.num_blocks) - 1,
            input_resolution: input_resolution,
            name: name
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state_before_downsample),
          attentions: Layers.append(state.attentions, attention)
        }
    end
  end

  defp stage(hidden_state, spec, opts) do
    name = opts[:name]
    downsample = opts[:downsample]
    hidden_size = opts[:hidden_size]
    num_blocks = opts[:num_blocks]
    num_attention_heads = opts[:num_attention_heads]
    input_resolution = opts[:input_resolution]

    # Note that we include only record hidden_state and attention
    # from the last block in each stage

    {hidden_state, attention} =
      for block_idx <- 0..(num_blocks - 1), reduce: {hidden_state, nil} do
        {hidden_state, _attention} ->
          name = name |> join("blocks") |> join(block_idx)

          shift_size =
            if rem(block_idx, 2) == 0 do
              0
            else
              div(spec.window_size, 2)
            end

          {hidden_state, attention} =
            transformer_block(hidden_state,
              num_attention_heads: num_attention_heads,
              hidden_size: hidden_size,
              kernel_initializer: kernel_initializer(spec),
              dropout_rate: spec.dropout_rate,
              attention_dropout_rate: spec.attention_dropout_rate,
              layer_norm_epsilon: spec.layer_norm_epsilon,
              intermediate_size: floor(spec.intermediate_size_ratio * hidden_size),
              activation: spec.activation,
              name: name,
              window_size: spec.window_size,
              shift_size: shift_size,
              input_resolution: input_resolution
            )

          {hidden_state, attention}
      end

    hidden_state_before_downsample = hidden_state

    hidden_state =
      if downsample do
        patch_merging(hidden_state,
          input_resolution: input_resolution,
          hidden_size: hidden_size,
          layer_norm_epsilon: spec.layer_norm_epsilon,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "downsample")
        )
      else
        hidden_state
      end

    {hidden_state, attention, hidden_state_before_downsample}
  end

  defp transformer_block(hidden_state, opts) do
    num_attention_heads = opts[:num_attention_heads]
    hidden_size = opts[:hidden_size]
    kernel_initializer = opts[:kernel_initializer]
    dropout_rate = opts[:dropout_rate]
    attention_dropout_rate = opts[:attention_dropout_rate]
    layer_norm_epsilon = opts[:layer_norm_epsilon]
    intermediate_size = opts[:intermediate_size]
    activation = opts[:activation]
    name = opts[:name]
    window_size = opts[:window_size]
    shift_size = opts[:shift_size]
    input_resolution = opts[:input_resolution]

    {shift_size, window_size} =
      if input_resolution <= window_size do
        {0, input_resolution}
      else
        {shift_size, window_size}
      end

    shortcut = hidden_state

    attention_mask =
      window_attention_mask(hidden_state, shift_size, window_size, input_resolution)

    relative_attention_bias =
      relative_attention_bias(window_size, num_attention_heads,
        name: join(name, "self_attention.relative_attention_bias")
      )

    hidden_state =
      hidden_state
      |> Axon.layer_norm(epsilon: layer_norm_epsilon, name: join(name, "self_attention_norm"))
      |> hidden_state_windows(shift_size, window_size, input_resolution)

    {hidden_state, attention, _self_attention_cache, _attention_relative_bias} =
      Layers.Transformer.multi_head_attention(hidden_state, hidden_state, hidden_state,
        attention_mask: attention_mask,
        attention_relative_bias: relative_attention_bias,
        num_heads: num_attention_heads,
        hidden_size: hidden_size,
        kernel_initializer: kernel_initializer,
        dropout_rate: attention_dropout_rate,
        name: join(name, "self_attention")
      )

    hidden_state =
      Axon.dropout(hidden_state, rate: dropout_rate, name: join(name, "self_attention_dropout"))

    hidden_state =
      hidden_state
      |> reverse_hidden_state_windows(shift_size, window_size, input_resolution)
      |> Axon.dropout(rate: dropout_rate)

    hidden_state = Axon.add(hidden_state, shortcut)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> Axon.layer_norm(epsilon: layer_norm_epsilon, name: join(name, "output_norm"))
      |> Axon.dense(intermediate_size, name: join(name, "ffn.intermediate"))
      |> Layers.activation(activation)
      |> Axon.dense(hidden_size, name: join(name, "ffn.output"))
      |> Axon.dropout(rate: dropout_rate)

    hidden_state = Axon.add(hidden_state, shortcut)

    {hidden_state, attention}
  end

  defp window_attention_mask(hidden_state, shift_size, window_size, input_resolution) do
    if shift_size > 0 do
      # Computes attention mask for shifted window multi-head self-attention (SW-MSA)

      Axon.nx(hidden_state, fn hidden_state ->
        {batch_size, _dimension, _hidden_size} = Nx.shape(hidden_state)
        height = width = input_resolution

        # See Figure 4. in the paper. We color the 2D patches (tokens)
        # into 4 groups. Then, we compute a mask such that each token
        # attends only to tokens within the same group.

        grid_0 = Nx.broadcast(0, {height - shift_size, width - shift_size})
        grid_b = Nx.broadcast(1, {height - shift_size, shift_size})
        grid_c = Nx.broadcast(2, {shift_size, width - shift_size})
        grid_a = Nx.broadcast(3, {shift_size, shift_size})

        grid =
          Nx.concatenate([
            Nx.concatenate([grid_0, grid_b], axis: 1),
            Nx.concatenate([grid_c, grid_a], axis: 1)
          ])

        windowed_patch_groups =
          grid
          |> Nx.reshape({1, height, width, 1})
          |> window_partition(window_size)
          |> Nx.reshape({:auto, window_size * window_size})

        windows_attention_mask =
          Nx.equal(
            Nx.new_axis(windowed_patch_groups, 1),
            Nx.new_axis(windowed_patch_groups, 2)
          )
          |> Nx.new_axis(1)

        # Note that we repeat the mask for each batched input, so that
        # the batch dimension has size batch_size * num_windows, which
        # matches the input. This way we can apply the mask as usual,
        # without reshaping back and forth.
        Nx.tile(windows_attention_mask, [batch_size, 1, 1, 1])
      end)
    else
      Layers.none()
    end
  end

  defp relative_attention_bias(window_size, num_attention_heads, opts) do
    name = opts[:name]

    kernel =
      Axon.param("kernel", {(2 * window_size - 1) * (2 * window_size - 1), num_attention_heads})

    Axon.layer(
      fn kernel, opts ->
        window_size = opts[:window_size]

        idx = relative_position_index(window_size) |> Nx.reshape({:auto})

        kernel
        |> Nx.take(idx)
        |> Nx.reshape({window_size * window_size, window_size * window_size, :auto})
        |> Nx.transpose(axes: [2, 0, 1])
        |> Nx.new_axis(0)
      end,
      [kernel],
      window_size: window_size,
      name: name
    )
  end

  defp relative_position_index(window_size) do
    coords_h = Nx.iota({window_size, window_size}, axis: 0) |> Nx.flatten()
    coords_w = Nx.iota({window_size, window_size}, axis: 1) |> Nx.flatten()
    coord_pairs = Nx.stack([coords_h, coords_w])

    relative_coords = Nx.subtract(Nx.new_axis(coord_pairs, 2), Nx.new_axis(coord_pairs, 1))

    relative_coords
    |> Nx.add(Nx.reshape(Nx.tensor([window_size - 1, window_size - 1]), {2, 1, 1}))
    |> Nx.multiply(Nx.reshape(Nx.tensor([2 * window_size - 1, 1]), {2, 1, 1}))
    |> Nx.sum(axes: [0])
  end

  defp hidden_state_windows(hidden_state, shift_size, window_size, input_resolution) do
    Axon.nx(hidden_state, fn hidden_state ->
      {batch_size, _dimension, hidden_size} = Nx.shape(hidden_state)

      height = width = input_resolution
      hidden_state = Nx.reshape(hidden_state, {batch_size, height, width, hidden_size})

      # Apply cyclic shift
      hidden_state =
        if shift_size > 0 do
          Bumblebee.Utils.Nx.roll(hidden_state, shifts: [-shift_size, -shift_size], axes: [1, 2])
        else
          hidden_state
        end

      # Partition windows
      hidden_state
      |> window_partition(window_size)
      |> Nx.reshape({:auto, window_size * window_size, hidden_size})
    end)
  end

  defp reverse_hidden_state_windows(hidden_state, shift_size, window_size, input_resolution) do
    Axon.nx(hidden_state, fn hidden_state ->
      {_batch_size, _dimension, hidden_size} = Nx.shape(hidden_state)
      height = width = input_resolution

      # Reverse window partitioning
      hidden_state =
        hidden_state
        |> Nx.reshape({:auto, window_size, window_size, hidden_size})
        |> window_unpartition(window_size, height, width)

      # Reverse cyclic shift
      hidden_state =
        if shift_size > 0 do
          Bumblebee.Utils.Nx.roll(hidden_state, shifts: [shift_size, shift_size], axes: [1, 2])
        else
          hidden_state
        end

      Nx.reshape(hidden_state, {:auto, height * width, hidden_size})
    end)
  end

  defnp window_partition(tensor, window_size) do
    {batch_size, height, width, hidden_size} = Nx.shape(tensor)
    windowed_height = div(height, window_size)
    windowed_width = div(width, window_size)

    Nx.reshape(
      tensor,
      {batch_size, windowed_height, window_size, windowed_width, window_size, hidden_size}
    )
    |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
    |> Nx.reshape({:auto, window_size, window_size, hidden_size})
  end

  defnp window_unpartition(tensor, window_size, height, width) do
    {_batch_size, _height, _width, hidden_size} = Nx.shape(tensor)
    windowed_height = div(height, window_size)
    windowed_width = div(width, window_size)

    Nx.reshape(
      tensor,
      {:auto, windowed_height, windowed_width, window_size, window_size, hidden_size}
    )
    |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
    |> Nx.reshape({:auto, height, width, hidden_size})
  end

  defp patch_merging(hidden_state, opts) do
    input_resolution = opts[:input_resolution]
    hidden_size = opts[:hidden_size]
    layer_norm_epsilon = opts[:layer_norm_epsilon]
    kernel_initializer = opts[:kernel_initializer]
    name = opts[:name]

    # We group patches from each 2x2 square and apply a dense layer
    # against each group

    hidden_state
    |> Axon.nx(fn hidden_state ->
      {batch_size, _sequence_length, _hidden_size} = Nx.shape(hidden_state)

      hidden_state =
        Nx.reshape(hidden_state, {batch_size, input_resolution, input_resolution, :auto})

      input_feature_0 = hidden_state[[.., 0..-1//2, 0..-1//2, ..]]
      input_feature_1 = hidden_state[[.., 1..-1//2, 0..-1//2, ..]]
      input_feature_2 = hidden_state[[.., 0..-1//2, 1..-1//2, ..]]
      input_feature_3 = hidden_state[[.., 1..-1//2, 1..-1//2, ..]]

      Nx.concatenate([input_feature_0, input_feature_1, input_feature_2, input_feature_3],
        axis: -1
      )
      |> Nx.reshape({batch_size, :auto, 4 * hidden_size})
    end)
    |> Axon.layer_norm(epsilon: layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dense(2 * hidden_size,
      kernel_initializer: kernel_initializer,
      name: join(name, "reduction"),
      use_bias: false
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          num_blocks: {"depths", list(number())},
          drop_path_rate: {"drop_path_rate", number()},
          embedding_size: {"embed_dim", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          image_size: {"image_size", number()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          intermediate_size_ratio: {"mlp_ratio", number()},
          num_channels: {"num_channels", number()},
          num_attention_heads: {"num_heads", list(number())},
          patch_size: {"patch_size", number()},
          use_attention_bias: {"qkv_bias", boolean()},
          use_absolute_position_embeddings: {"use_absolute_embeddings", boolean()},
          window_size: {"window_size", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding.projection" => "swin.embeddings.patch_embeddings.projection",
        "embedder.norm" => "swin.embeddings.norm",
        "encoder.stages.{n}.blocks.{m}.output_norm" =>
          "swin.encoder.layers.{n}.blocks.{m}.layernorm_after",
        "encoder.stages.{n}.blocks.{m}.self_attention_norm" =>
          "swin.encoder.layers.{n}.blocks.{m}.layernorm_before",
        "encoder.stages.{n}.blocks.{m}.self_attention.key" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.self.key",
        "encoder.stages.{n}.blocks.{m}.self_attention.output" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.output.dense",
        "encoder.stages.{n}.blocks.{m}.self_attention.query" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.self.query",
        "encoder.stages.{n}.blocks.{m}.self_attention.value" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.self.value",
        "encoder.stages.{n}.blocks.{m}.self_attention.relative_attention_bias" => %{
          "kernel" => {
            [
              {"swin.encoder.layers.{n}.blocks.{m}.attention.self",
               "relative_position_bias_table"}
            ],
            fn [kernel] -> kernel end
          }
        },
        "encoder.stages.{n}.blocks.{m}.ffn.intermediate" =>
          "swin.encoder.layers.{n}.blocks.{m}.intermediate.dense",
        "encoder.stages.{n}.blocks.{m}.ffn.output" =>
          "swin.encoder.layers.{n}.blocks.{m}.output.dense",
        "encoder.stages.{n}.downsample.norm" => "swin.encoder.layers.{n}.downsample.norm",
        "encoder.stages.{n}.downsample.reduction" =>
          "swin.encoder.layers.{n}.downsample.reduction",
        "norm" => "swin.layernorm",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end
