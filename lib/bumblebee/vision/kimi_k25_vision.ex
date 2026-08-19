defmodule Bumblebee.Vision.KimiK25Vision do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: {448, 448},
        doc: """
        the size of the input image. The upstream model processes images at their native
        resolution, however a computation graph requires a fixed shape, so the images are
        resized to this size
        """
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 14,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 1152,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 4304,
        doc: "the dimensionality of intermediate layers"
      ],
      num_blocks: [
        default: 27,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      position_embedding_size: [
        default: 64,
        doc: """
        the size of the learnt position embedding grid. The grid is resampled to the patch grid
        of the input image
        """
      ],
      merge_size: [
        default: {2, 2},
        doc: "the size of the patch block that is pooled into a single output token"
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++ Shared.common_options([:output_hidden_states, :output_attentions])

  @moduledoc """
  The vision model of the Kimi K2.5 model family.

  The model is a vision Transformer with two-dimensional rotary embedding
  and a learnt position embedding grid, which is resampled to the patch
  grid of the input image. The output patches are pooled in blocks of
  `:merge_size`, which is what the multimodal model projects into the
  text embedding space.

  ## Architectures

    * `:base` - the vision model without any head on top

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Nx.Defn
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
    %{"pixel_values" => Nx.template({1, image_height(spec), image_width(spec), 3}, :f32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    shape = {nil, image_height(spec), image_width(spec), spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([Axon.input("pixel_values", shape: shape)])
  end

  defp core(inputs, spec) do
    {grid_height, grid_width} = patch_grid(spec)

    hidden_state = embedder(inputs["pixel_values"], spec, name: "embedder")

    encoder_outputs =
      encoder(hidden_state, spec, name: "encoder")

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: "output_norm"
      )

    {merge_height, merge_width} = spec.merge_size

    # The patches are pooled in blocks, which are the tokens that the
    # multimodal model splices into the text sequence
    pooled_state =
      Axon.nx(hidden_state, fn hidden_state ->
        batch_size = Nx.axis_size(hidden_state, 0)
        hidden_size = Nx.axis_size(hidden_state, 2)

        hidden_state
        |> Nx.reshape(
          {batch_size, div(grid_height, merge_height), merge_height, div(grid_width, merge_width),
           merge_width, hidden_size}
        )
        |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
        |> Nx.reshape(
          {batch_size, div(grid_height, merge_height) * div(grid_width, merge_width),
           merge_height * merge_width, hidden_size}
        )
      end)

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    {grid_height, grid_width} = patch_grid(spec)

    patch_embeddings =
      pixel_values
      |> Axon.conv(spec.hidden_size,
        kernel_size: spec.patch_size,
        strides: spec.patch_size,
        padding: :valid,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "patch_embedding")
      )
      |> Axon.reshape({:batch, grid_height * grid_width, spec.hidden_size})

    # The learnt position embedding grid is resampled to the patch grid
    # with bicubic interpolation. The interpolation only depends on the
    # two grid sizes, so it is a constant linear map
    interpolation =
      interpolation_matrix(grid_height, grid_width, spec.position_embedding_size)

    position_embedding =
      Axon.param(
        "embedding",
        fn _ ->
          {spec.position_embedding_size, spec.position_embedding_size, spec.hidden_size}
        end,
        initializer: :zeros
      )

    position_embeddings =
      Axon.layer(&interpolate_position_embedding/3, [patch_embeddings, position_embedding],
        name: join(name, "position_embedding"),
        op_name: :position_embedding,
        interpolation: interpolation
      )

    Axon.add(patch_embeddings, position_embeddings)
  end

  defnp interpolate_position_embedding(_patch_embeddings, embedding, opts \\ []) do
    opts = keyword!(opts, [:interpolation, mode: :inference])

    hidden_size = Nx.axis_size(embedding, 2)

    embedding
    |> Nx.reshape({:auto, hidden_size})
    |> then(&Nx.dot(Nx.as_type(opts[:interpolation], Nx.type(&1)), &1))
    |> Nx.new_axis(0)
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    {cos, sin} = rotary_embedding(spec)

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      layer_norm: &Axon.layer_norm(&1, epsilon: spec.layer_norm_epsilon, name: &2),
      block_type: :norm_first,
      attention: &attention(&1, &2, spec, cos, sin),
      ffn: [intermediate_size: spec.intermediate_size, activation: spec.activation],
      name: join(name, "blocks")
    )
  end

  # The attention is the standard multi-head attention with a
  # two-dimensional rotary embedding, computed from the position of each
  # patch in the grid
  defp attention(hidden_state, opts, spec, cos, sin) do
    name = opts[:name]

    num_heads = spec.num_attention_heads
    head_size = div(spec.hidden_size, num_heads)

    [query, key, value] =
      for {kind, layer_name} <- [{:query, "query"}, {:key, "key"}, {:value, "value"}] do
        projection =
          hidden_state
          |> Axon.dense(num_heads * head_size,
            kernel_initializer: kernel_initializer(spec),
            name: join(name, layer_name)
          )
          |> Layers.split_heads(num_heads)

        if kind == :value do
          projection
        else
          Axon.nx(projection, &rotate(&1, cos, sin))
        end
      end

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        Layers.none(),
        opts[:attention_head_mask],
        Layers.none(),
        Layers.none(),
        causal: false
      )

    attention_output =
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output")
      )

    {attention_output, attention_weights, Layers.none(), Layers.none()}
  end

  defp rotate(hidden_state, cos, sin) do
    cos = Nx.as_type(cos, :f32)
    sin = Nx.as_type(sin, :f32)

    type = Nx.type(hidden_state)
    hidden_state = Nx.as_type(hidden_state, :f32)

    size = div(Nx.axis_size(hidden_state, -1), 2)
    first = Nx.slice_along_axis(hidden_state, 0, size, axis: -1)
    second = Nx.slice_along_axis(hidden_state, size, size, axis: -1)
    rotated = Nx.concatenate([Nx.negate(second), first], axis: -1)

    hidden_state
    |> Nx.multiply(cos)
    |> Nx.add(Nx.multiply(rotated, sin))
    |> Nx.as_type(type)
  end

  # The rotary frequencies depend only on the patch grid, so they are
  # constant. Each patch is rotated by its position along both grid axes
  defp rotary_embedding(spec) do
    {grid_height, grid_width} = patch_grid(spec)

    head_size = div(spec.hidden_size, spec.num_attention_heads)
    spatial_size = div(head_size, 2)

    inv_frequency =
      Nx.iota({div(spatial_size, 2)})
      |> Nx.multiply(2)
      |> Nx.divide(spatial_size)
      |> then(&Nx.divide(1.0, Nx.pow(spec.rotary_embedding_base, &1)))

    rows = Nx.iota({grid_height, grid_width}, axis: 0) |> Nx.reshape({:auto, 1})
    columns = Nx.iota({grid_height, grid_width}, axis: 1) |> Nx.reshape({:auto, 1})

    # Note that the upstream implementation flips the axes, so that the
    # column position comes first
    frequencies =
      [
        Nx.new_axis(Nx.multiply(columns, inv_frequency), -1),
        Nx.new_axis(Nx.multiply(rows, inv_frequency), -1)
      ]
      |> Nx.concatenate(axis: -1)
      |> Nx.reshape({grid_height * grid_width, :auto})

    angle = Nx.concatenate([frequencies, frequencies], axis: -1)

    cos = angle |> Nx.cos() |> Nx.new_axis(0) |> Nx.new_axis(2)
    sin = angle |> Nx.sin() |> Nx.new_axis(0) |> Nx.new_axis(2)

    {cos, sin}
  end

  # Bicubic resampling of a square grid of `side` positions to a grid of
  # `height` by `width` positions, as a dense matrix that maps the
  # flattened source grid to the flattened target grid
  defp interpolation_matrix(height, width, side) do
    {row_taps, row_weights} = axis_taps(height, side)
    {column_taps, column_weights} = axis_taps(width, side)

    taps = Nx.axis_size(row_taps, 1)
    num_patches = height * width

    # The interpolation is separable, so each patch gathers every pair
    # of a row tap and a column tap
    indices =
      row_taps
      |> Nx.reshape({height, 1, taps, 1})
      |> Nx.multiply(side)
      |> Nx.add(Nx.reshape(column_taps, {1, width, 1, taps}))
      |> Nx.reshape({num_patches * taps * taps, 1})

    weights =
      row_weights
      |> Nx.reshape({height, 1, taps, 1})
      |> Nx.multiply(Nx.reshape(column_weights, {1, width, 1, taps}))
      |> Nx.reshape({num_patches * taps * taps})

    patches =
      Nx.iota({num_patches, taps * taps}, axis: 0)
      |> Nx.reshape({num_patches * taps * taps, 1})

    Nx.indexed_add(
      Nx.broadcast(0.0, {num_patches, side * side}),
      Nx.concatenate([patches, indices], axis: 1),
      weights
    )
  end

  defp axis_taps(size, side) do
    index = Nx.iota({size, 1}) |> Nx.as_type(:f32)

    source = Nx.subtract(Nx.divide(Nx.multiply(Nx.add(index, 0.5), side), size), 0.5)
    floor = Nx.floor(source)

    offsets = Nx.reshape(Nx.subtract(Nx.iota({4}), 1), {1, 4})

    taps =
      floor
      |> Nx.as_type(:s64)
      |> Nx.add(offsets)
      |> Nx.clip(0, side - 1)

    distance = Nx.abs(Nx.subtract(Nx.subtract(source, floor), offsets))

    a = -0.75

    near =
      Nx.add(
        Nx.multiply(
          Nx.multiply(Nx.subtract(Nx.multiply(a + 2, distance), a + 3), distance),
          distance
        ),
        1
      )

    far =
      Nx.subtract(
        Nx.multiply(
          Nx.add(Nx.multiply(Nx.subtract(Nx.multiply(a, distance), 5 * a), distance), 8 * a),
          distance
        ),
        4 * a
      )

    weights = Nx.select(Nx.less_equal(distance, 1), near, far)

    {taps, weights}
  end

  defp patch_grid(spec) do
    {div(image_height(spec), spec.patch_size), div(image_width(spec), spec.patch_size)}
  end

  defp image_height(%{image_size: {height, _width}}), do: height
  defp image_width(%{image_size: {_height, width}}), do: width

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      data = data |> vision_config() |> Shared.normalize_rope_options()

      opts =
        convert!(data,
          patch_size: {"patch_size", number()},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          position_embedding_size: {"pos_emb_height", number()},
          merge_size: {"merge_kernel_size", tuple([number(), number()])},
          activation: {"hidden_act", activation()},
          rotary_embedding_base: {"rope_theta", number()}
        )

      @for.config(spec, opts)
    end

    defp vision_config(%{"vision_config" => %{} = vision_config}), do: vision_config
    defp vision_config(data), do: data
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    # The checkpoints use the naming of the original implementation,
    # where the query, key and value projections are a single tensor
    def params_mapping(spec) do
      %{
        "embedder.patch_embedding" => "encoder.patch_embed.proj",
        "embedder.position_embedding" => %{
          "embedding" => {
            [{"patch_embed.pos_emb", "weight"}],
            fn [embedding] -> embedding end
          }
        },
        "encoder.blocks.{n}.self_attention.output" => "encoder.blocks.{n}.wo",
        "encoder.blocks.{n}.self_attention_norm" => "encoder.blocks.{n}.norm0",
        "encoder.blocks.{n}.ffn.intermediate" => "encoder.blocks.{n}.mlp.fc0",
        "encoder.blocks.{n}.ffn.output" => "encoder.blocks.{n}.mlp.fc1",
        "encoder.blocks.{n}.output_norm" => "encoder.blocks.{n}.norm1",
        "output_norm" => "encoder.final_layernorm"
      }
      |> Map.merge(attention_params(spec))
    end

    defp attention_params(spec) do
      for {name, index} <- [{"query", 0}, {"key", 1}, {"value", 2}], into: %{} do
        {"encoder.blocks.{n}.self_attention.#{name}",
         %{
           "kernel" => {
             [{"encoder.blocks.{n}.wqkv", "weight"}],
             fn [kernel] ->
               kernel
               |> chunk(index)
               |> maybe_permute(name, spec)
               |> Nx.transpose()
             end
           },
           "bias" => {
             [{"encoder.blocks.{n}.wqkv", "bias"}],
             fn [bias] -> bias |> chunk(index) |> maybe_permute(name, spec) end
           }
         }}
      end
    end

    defp chunk(tensor, index) do
      size = div(Nx.axis_size(tensor, 0), 3)
      Nx.slice_along_axis(tensor, index * size, size, axis: 0)
    end

    # The original implementation lays the rotary dimensions out as
    # interleaved pairs, so the query and key projections are permuted
    # into the two-halves layout
    defp maybe_permute(tensor, "value", _spec), do: tensor

    defp maybe_permute(tensor, _name, spec) do
      num_heads = spec.num_attention_heads
      size = Nx.axis_size(tensor, 0)
      half_head_size = div(div(size, num_heads), 2)

      trailing = Nx.shape(tensor) |> Tuple.to_list() |> Enum.drop(1)

      tensor
      |> Nx.reshape(List.to_tuple([num_heads, half_head_size, 2] ++ trailing))
      |> Nx.transpose(axes: [0, 2, 1] ++ Enum.to_list(3..(2 + length(trailing))//1))
      |> Nx.reshape(List.to_tuple([size] ++ trailing))
    end
  end
end
