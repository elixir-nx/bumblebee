defmodule Bumblebee.Vision.Swin do
  alias Bumblebee.Shared

  options =
    [
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      depths: [
        default: [2, 2, 18, 2],
        doc: "the depth (number of residual blocks) at each stage"
      ],
      drop_path_rate: [
        default: 0.1,
        doc: "the drop path rate used to for stochastic depth"
      ],
      # Maybe it should be renamed to hidden_size
      embed_dim: [
        default: 128,
        doc: ""
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder and decoder"
      ],
      image_size: [
        default: 384,
        doc: "the size of the input spatial dimensions"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      intermediate_size_ratio: [
        default: 4,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder,
        expressed as a multiplier of `:hidden_size`
        """
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      num_heads: [
        default: [4, 8, 16, 32],
        doc: "number of attention heads"
      ],
      patch_size: [
        default: 4,
        doc: "the size of the patch spatial dimensions"
      ],
      path_norm: [
        default: true,
        doc: ""
      ],
      use_attention_bias: [
        default: true,
        doc: "whether to use bias in query, key, and value projections"
      ],
      use_absolute_embeddings: [
        default: false,
        doc: ""
      ],
      window_size: [
        default: 12,
        doc: ""
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  Swin Transformer model.

  ## Architectures

    * `:for_image_classification` - Swin tranformer model for image classification.

  ## Global layer options

  # {Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  # {Shared.options_doc(options)}

  ## References

    * [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:for_image_classification]

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
      outputs.hidden_state
      |> Layers.take_token(index: 0, axis: 1)
      |> Axon.dense(spec.num_labels,
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

  # Contrary to Python implementation we do not have here argument
  # bool_maked_pos. This parameter is propagated from model through
  # core to embedder.
  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["pixel_values"], spec, name: join(name, "embedder"))

    {hidden_state, hidden_states, attentions} =
      encoder(embeddings, spec, name: join(name, "encoder"))

    hidden_state =
      Axon.layer_norm(hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "layernorm")
      )

    pooled_state =
      Axon.adaptive_avg_pool(hidden_state, output_size: {1, 1}, name: join(name, "pooler"))

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: hidden_states,
      attentions: attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    embeddings =
      pixel_values
      |> patch_embedding(spec, name: join(name, "patch_embedding"))
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon)

    embeddings =
      if spec.use_absolute_embeddings do
        num_patches = div(spec.image_size, spec.patch_size) ** 2

        position_embeddings =
          Layers.learned_embeddings(num_patches, spec.embed_dim,
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
    hidden_size = spec.embed_dim

    pixel_values
    |> Axon.conv(hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection")
    )
    |> Axon.reshape({:batch, :auto, spec.embed_dim}, name: join(name, "reshape"))
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]
    hidden_states = Axon.container({hidden_state})
    attentions = Axon.container({})

    state = {
      hidden_state,
      hidden_states,
      attentions
    }

    for stage_idx <- 0..(length(spec.depths) - 1), reduce: state do
      {hidden_state, hidden_states, attentions} ->
        {hidden_state, attention} =
          stage(hidden_state, stage_idx, spec, join("#{name}.blocks", stage_idx))

        {
          hidden_state,
          Layers.append(hidden_states, hidden_state),
          Layers.append(attentions, attention)
        }
    end
  end

  defp stage(hidden_state, stage_idx, spec, name) do
    grid_size = div(spec.image_size, spec.patch_size)
    input_resolution = div(grid_size, 2 ** stage_idx)
    dim = spec.embed_dim * 2 ** stage_idx
    num_attention_heads = Enum.at(spec.num_heads, stage_idx)

    {hidden_state, attention} =
      for layer_idx <- 0..(Enum.at(spec.depths, stage_idx) - 1), reduce: {hidden_state, nil} do
        {hidden_state, _} ->
          {hidden_state, attention} =
            layer(hidden_state, layer_idx, dim, num_attention_heads, spec, name)

          {hidden_state, attention}
      end

    hidden_state =
      if stage_idx < length(spec.depths) - 1 do
        downsample(hidden_state, input_resolution, dim, spec.layer_norm_epsilon, name)
      else
        hidden_state
      end

    {hidden_state, attention}
  end

  defp layer(hidden_state, layer_idx, dim, num_attention_heads, spec, name) do
    shortcut = hidden_state
    attn_mask = attention_mask_layer(hidden_state, layer_idx, spec)
    name = join(name, "layer.#{layer_idx}")

    {hidden_state, attention} =
      hidden_state
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "layernorm_before"))
      |> reshape(layer_idx)
      |> hidden_windows(layer_idx, spec)
      |> attention(attn_mask, num_attention_heads, dim, spec, name)

    hidden_state =
      {hidden_state, shortcut}
      |> unroll(layer_idx, spec)
      |> Axon.dropout(rate: spec.dropout_rate)

    output =
      Axon.add(shortcut, hidden_state)
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "layernorm_after"))
      |> Axon.dense(dim, name: join(name, "dense"))
      |> Layers.activation(spec.activation)
      |> Axon.dense(dim, name: join(name, "dense"))
      |> Axon.dropout(rate: spec.dropout_rate)

    hidden_state = Axon.add(hidden_state, output)

    {hidden_state, attention}
  end

  defp reshape(input, layer_idx) do
    input
    |> Axon.nx(
      fn x ->
        {batch_size, dimension, num_channels} = Nx.shape(x)
        height_width = dimension |> :math.sqrt() |> floor()

        x
        |> Nx.reshape({batch_size, height_width, height_width, num_channels})
      end,
      name: "reshape_#{layer_idx}"
    )
  end

  defp hidden_windows(input, layer_idx, spec) do
    shift_size = if 0 == rem(layer_idx, 2), do: 0, else: div(spec.window_size, 2)

    input
    |> Axon.nx(
      fn x ->
        {_batch_size, height, width, num_channels} = Nx.shape(x)

        {shift_size, _window_size} =
          if min(height, width) <= spec.window_size,
            do: {0, min(height, width)},
            else: {shift_size, spec.window_size}

        shiffted_hidden_state =
          if shift_size > 0,
            do: roll(x, shifts: [-shift_size, -shift_size], axes: [1, 2]),
            else: x

        shiffted_hidden_state
        |> window_partition(spec.window_size)
        |> Nx.reshape({:auto, spec.window_size * spec.window_size, num_channels})
      end,
      name: "hidden_windows_#{layer_idx}"
    )
  end

  defp attention(input, attention_mask, num_attention_heads, dim, spec, name) do
    {hidden_state, attention, _self_attention_cache, _attention_relative_bias} =
      Bumblebee.Layers.Transformer.multi_head_attention(
        input,
        input,
        input,
        attention_mask: attention_mask,
        num_heads: num_attention_heads,
        hidden_size: dim,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.attention_dropout_rate,
        name: join(name, "self_attention")
      )

    hidden_state =
      Axon.dropout(hidden_state,
        rate: spec.dropout_rate,
        name: join(name, "self_attention_dropout")
      )

    {hidden_state, attention}
  end

  def unroll({hidden_state, input}, layer_idx, spec) do
    shift_size = if 0 == rem(layer_idx, 2), do: 0, else: div(spec.window_size, 2)

    Axon.layer(
      fn state, input, _ ->
        {batch_size, dimension, num_channels} = Nx.shape(input)
        height_width = dimension |> :math.sqrt() |> floor()

        {shift_size, window_size} =
          if height_width <= spec.window_size,
            do: {0, height_width},
            else: {shift_size, spec.window_size}

        shifted_windows =
          state
          |> Nx.reshape({:auto, window_size, window_size, num_channels})
          |> window_reverse(window_size)

        if shift_size > 0 do
          roll(shifted_windows, shifts: [shift_size, shift_size], axes: [1, 2])
          |> Nx.reshape({batch_size, height_width * height_width, num_channels})
        else
          shifted_windows
          |> Nx.reshape({batch_size, height_width * height_width, num_channels})
        end
      end,
      [hidden_state, input],
      name: "unroll_#{layer_idx}"
    )
  end

  defp attention_mask_layer(hidden_state, layer_idx, spec) do
    shift_size = if 0 == rem(layer_idx, 2), do: 0, else: div(spec.window_size, 2)

    hidden_state
    |> Axon.nx(
      fn x ->
        {_batch_size, dimension, _num_channels} = Nx.shape(x)
        height_width = dimension |> :math.sqrt() |> floor()

        {shift_size, window_size} =
          if height_width <= spec.window_size,
            do: {0, height_width},
            else: {shift_size, spec.window_size}

        attention_mask(height_width, height_width, window_size, shift_size)
      end,
      name: "att_mask_#{layer_idx}"
    )
  end

  def attention_mask(height, width, window_size, shift_size) do
    if shift_size > 0 do
      # calculate attention mask for shifted window multi-head self-attention (SW-MSA)
      img_mask = Nx.broadcast(0.0, {1, height, width, 1})

      hslices = [
        0..(height - window_size - 1),
        (height - window_size)..(height - shift_size - 1),
        (height - shift_size)..(height - 1)
      ]

      wslices = [
        0..(width - window_size - 1),
        (width - window_size)..(width - shift_size - 1),
        (width - shift_size)..(width - 1)
      ]

      {img_mask, _count} =
        for hrange <- hslices, wrange <- wslices, reduce: {img_mask, 0.0} do
          {mask, count} ->
            mask =
              for hidx <- hrange, widx <- wrange, reduce: mask do
                deepest_mask ->
                  Nx.indexed_put(deepest_mask, Nx.tensor([0, hidx, widx, 0]), count)
              end

            {mask, count + 1.0}
        end

      mask_windows =
        img_mask
        |> window_partition(window_size)
        |> Nx.reshape({:auto, window_size * window_size})

      mask_windows
      |> Nx.new_axis(1)
      |> Nx.subtract(Nx.new_axis(mask_windows, 2))
      |> Nx.equal(0)
      |> Nx.logical_not()
    else
      %Axon.None{}
    end
  end

  defp window_partition(%Axon{} = input_feature, window_size) do
    input_feature
    |> Axon.nx(fn x -> window_partition(x, window_size) end)
  end

  defp window_partition(%Nx.Tensor{} = tensor, window_size) do
    {batch_size, height, width, num_channels} = Nx.shape(tensor)
    windowed_height = div(height, window_size)
    windowed_width = div(width, window_size)

    Nx.reshape(
      tensor,
      {batch_size, windowed_height, window_size, windowed_width, window_size, num_channels}
    )
    |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
    |> Nx.reshape({:auto, window_size, window_size, num_channels})
  end

  defp window_reverse(%Axon{} = input_feature, window_size) do
    input_feature
    |> Axon.nx(fn x -> window_reverse(x, window_size) end)
  end

  defp window_reverse(%Nx.Tensor{} = tensor, window_size) do
    {_batch_size, height, width, num_channels} = Nx.shape(tensor)
    windowed_height = div(height, window_size)
    windowed_width = div(width, window_size)

    Nx.reshape(
      tensor,
      {:auto, windowed_height, windowed_width, window_size, window_size, num_channels}
    )
    |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
    |> Nx.reshape({:auto, height, width, num_channels})
  end

  defp downsample(hidden_state, input_resolution, dim, norm_epsilon, name) do
    Axon.nx(hidden_state, fn x ->
      {batch_size, _dim, num_channels} = Nx.shape(x)

      x = Nx.reshape(x, {batch_size, input_resolution, input_resolution, :auto})

      input_feature_0 = x[[.., 0..-1//2, 0..-1//2, ..]]
      input_feature_1 = x[[.., 1..-1//2, 0..-1//2, ..]]
      input_feature_2 = x[[.., 0..-1//2, 1..-1//2, ..]]
      input_feature_3 = x[[.., 1..-1//2, 1..-1//2, ..]]

      Nx.concatenate([input_feature_0, input_feature_1, input_feature_2, input_feature_3],
        axis: -1
      )
      |> Nx.reshape({batch_size, :auto, 4 * num_channels})
    end)
    |> Axon.layer_norm(epsilon: norm_epsilon, name: join(name, "downsample_norm"))
    |> Axon.dense(2 * dim,
      kernel_initializer: Axon.Initializers.uniform(),
      name: join(name, "downsample_reduction")
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
          depths: {"depths", list(number())},
          drop_path_rate: {"drop_path_rate", number()},
          embed_dim: {"embed_dim", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          image_size: {"image_size", number()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          intermediate_size_ratio: {"mlp_ratio", number()},
          num_channels: {"num_channels", number()},
          num_heads: {"num_heads", list(number())},
          patch_size: {"patch_size", number()},
          path_norm: {"path_norm", boolean()},
          use_attention_bias: {"qkv_bias", boolean()},
          use_absolute_embeddings: {"use_absolute_embeddings", boolean()},
          window_size: {"window_size", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "layernorm" => "swin.layernorm",
        "encoder.blocks.{n}.layer.{m}.self_attention.output" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.output.dense",
        "encoder.blocks.{n}.layer.{m}.self_attention.value" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.self.value",
        "encoder.blocks.{n}.layer.{m}.self_attention.query" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.self.query",
        "encoder.blocks.{n}.layer.{m}.self_attention.key" =>
          "swin.encoder.layers.{n}.blocks.{m}.attention.self.key",
        "encoder.blocks.{n}.layer.{m}.layernorm_before" =>
          "swin.encoder.layers.{n}.blocks.{m}.layernorm_before",
        "encoder.blocks.{n}.layer.{m}.layernorm_after" =>
          "swin.encoder.layers.{n}.blocks.{m}.layernorm_after",
        "embedder.patch_embedding.projection" => "swin.embeddings.patch_embeddings.projection",
        "encoder.blocks.{n}.downsample_norm" => "swin.encoder.layers.{n}.downsample.norm",
        "encoder.blocks.{n}.downsample_reduction" =>
          "swin.encoder.layers.{n}.downsample.reduction",
        "image_classification_head.output" => "classifier",
        "encoder.blocks.{n}.layer.{m}.dense" => "swin.encoder.layers.{n}.blocks.{m}.output.dense"
      }
    end
  end

  defp roll(%Axon{} = x, opts) do
    Axon.nx(x, fn y -> roll(y, opts) end)
  end

  defp roll(%Nx.Tensor{} = x, opts) do
    opts = Keyword.validate!(opts, shifts: [], axes: [])
    shifts = opts[:shifts]
    axes = opts[:axes]

    if length(shifts) != length(axes) do
      raise ArgumentError, "shifts and axes must align, shifts: #{shifts}, axes: #{axes}"
    else
      shape = Nx.shape(x) |> Tuple.to_list()

      Enum.zip(shifts, axes)
      |> Enum.reduce(x, fn {shift, dim}, acc ->
        shift = rem(shift, Enum.at(shape, dim))

        if 0 < shift do
          {base, move} = Nx.split(acc, -1 * shift, axis: dim)
          Nx.concatenate([move, base], axis: dim)
        else
          acc
        end
      end)
    end
  end
end
