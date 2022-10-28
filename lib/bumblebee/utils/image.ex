defmodule Bumblebee.Utils.Image do
  @moduledoc false

  import Nx.Defn

  @compile {:no_warn_undefined, StbImage}

  @doc """
  Converts the given term to a batch of images.
  """
  def to_batched_tensor(%Nx.Tensor{} = image) do
    case Nx.rank(image) do
      3 ->
        Nx.new_axis(image, 0, :batch)

      4 ->
        image

      rank ->
        raise ArgumentError,
              "expected image to be a rank-3 image or a rank-4 batch, got rank: #{rank}"
    end
  end

  def to_batched_tensor(image) when is_struct(image, StbImage) do
    image
    |> StbImage.to_nx()
    |> Nx.new_axis(0, :batch)
  end

  @doc """
  Crops an image at the center.

  If the image is too small to be cropped to the desired size, it gets
  padded with zeros.

  ## Options

    * `:size` - the target image size specified as `{height, width}`

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  ## Examples

      iex> images = Nx.iota({1, 4, 4, 1})
      iex> Bumblebee.Utils.Image.center_crop(images, size: {2, 2})
      #Nx.Tensor<
        s64[1][2][2][1]
        [
          [
            [
              [5],
              [6]
            ],
            [
              [9],
              [10]
            ]
          ]
        ]
      >

      iex> images = Nx.iota({1, 2, 2, 1})
      iex> Bumblebee.Utils.Image.center_crop(images, size: {1, 4})
      #Nx.Tensor<
        s64[1][1][4][1]
        [
          [
            [
              [0],
              [0],
              [1],
              [0]
            ]
          ]
        ]
      >

  """
  defn center_crop(input, opts \\ []) do
    opts = keyword!(opts, [:size, channels: :last])

    pad_config =
      transform({input, opts}, fn {input, opts} ->
        for {axis, size, out_size} <- spatial_axes_with_sizes(input, opts),
            reduce: List.duplicate({0, 0, 0}, Nx.rank(input)) do
          pad_config ->
            low = div(size - out_size, 2)
            high = low + out_size
            List.replace_at(pad_config, axis, {-low, high - size, 0})
        end
      end)

    Nx.pad(input, 0, pad_config)
  end

  defnp spatial_axes_with_sizes(input, opts \\ []) do
    {height_axis, width_axis} = spatial_axes(input, channels: opts[:channels])
    {height, width} = size(input, channels: opts[:channels])
    {out_height, out_width} = opts[:size]
    [{height_axis, height, out_height}, {width_axis, width, out_width}]
  end

  @doc """
  Resizes an image.

  ## Options

    * `:size` - the target image size specified as `{height, width}`

    * `:method` - the resizing method to use, either of `:nearest`,
      `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`. Defaults to
      `:bilinear`

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  ## Examples

      iex> images = Nx.iota({1, 2, 2, 1}, type: {:f, 32})
      iex> Bumblebee.Utils.Image.resize(images, size: {3, 3}, method: :nearest)
      #Nx.Tensor<
        f32[1][3][3][1]
        [
          [
            [
              [0.0],
              [1.0],
              [1.0]
            ],
            [
              [2.0],
              [3.0],
              [3.0]
            ],
            [
              [2.0],
              [3.0],
              [3.0]
            ]
          ]
        ]
      >

      iex> images = Nx.iota({1, 2, 2, 1}, type: {:f, 32})
      iex> Bumblebee.Utils.Image.resize(images, size: {3, 3}, method: :bilinear)
      #Nx.Tensor<
        f32[1][3][3][1]
        [
          [
            [
              [0.0],
              [0.5],
              [1.0]
            ],
            [
              [1.0],
              [1.5],
              [2.0]
            ],
            [
              [2.0],
              [2.5],
              [3.0]
            ]
          ]
        ]
      >

  """
  defn resize(input, opts \\ []) do
    opts = keyword!(opts, [:size, channels: :last, method: :bilinear])

    transform({input, opts}, fn {input, opts} ->
      {spatial_axes, out_shape} =
        input
        |> spatial_axes_with_sizes(opts)
        |> Enum.reject(fn {_axis, size, out_size} -> Elixir.Kernel.==(size, out_size) end)
        |> Enum.map_reduce(Nx.shape(input), fn {axis, _size, out_size}, out_shape ->
          {axis, put_elem(out_shape, axis, out_size)}
        end)

      resized_input =
        case opts[:method] do
          :nearest ->
            resize_nearest(input, out_shape, spatial_axes)

          :bilinear ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_linear_kernel/1)

          :bicubic ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_cubic_kernel/1)

          :lanczos3 ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_lanczos_kernel(3, &1))

          :lanczos5 ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_lanczos_kernel(5, &1))

          method ->
            raise ArgumentError,
                  "expected :method to be either of :nearest, :bilinear, :bicubic, " <>
                    ":lanczos3, :lanczos5, got: #{inspect(method)}"
        end

      cast_to(resized_input, input)
    end)
  end

  defnp spatial_axes(input, opts \\ []) do
    channels = opts[:channels]

    transform({input, channels}, fn {input, channels} ->
      axes =
        case channels do
          :first -> [-2, -1]
          :last -> [-3, -2]
        end

      axes
      |> Enum.map(&Nx.axis_index(input, &1))
      |> List.to_tuple()
    end)
  end

  defnp cast_to(left, right) do
    left_type = Nx.type(left)
    right_type = Nx.type(right)

    left =
      if Nx.Type.float?(left_type) and Nx.Type.integer?(right_type) do
        Nx.round(left)
      else
        left
      end

    left
    |> Nx.as_type(right_type)
    |> Nx.reshape(left, names: Nx.names(right))
  end

  defnp resize_nearest(input, out_shape, spatial_axes) do
    transform({input, out_shape, spatial_axes}, fn {input, out_shape, spatial_axes} ->
      singular_shape = List.duplicate(1, Nx.rank(input)) |> List.to_tuple()

      for axis <- spatial_axes, reduce: input do
        input ->
          input_shape = Nx.shape(input)
          input_size = elem(input_shape, axis)
          output_size = elem(out_shape, axis)
          inv_scale = input_size / output_size
          offset = (Nx.iota({output_size}) + 0.5) * inv_scale
          offset = offset |> Nx.floor() |> Nx.as_type({:s, 32})

          offset =
            offset
            |> Nx.reshape(put_elem(singular_shape, axis, output_size))
            |> Nx.broadcast(put_elem(input_shape, axis, output_size))

          Nx.take_along_axis(input, offset, axis: axis)
      end
    end)
  end

  @f32_eps :math.pow(2, -23)

  defnp resize_with_kernel(input, out_shape, spatial_axes, kernel_fun) do
    transform({input, out_shape, spatial_axes}, fn {input, out_shape, spatial_axes} ->
      for axis <- spatial_axes, reduce: input do
        input ->
          input_shape = Nx.shape(input)
          input_size = elem(input_shape, axis)
          output_size = elem(out_shape, axis)

          inv_scale = input_size / output_size
          kernel_scale = Nx.max(1, inv_scale)

          sample_f = (Nx.iota({1, output_size}) + 0.5) * inv_scale - 0.5
          x = Nx.abs(sample_f - Nx.iota({input_size, 1})) / kernel_scale
          weights = kernel_fun.(x)

          weights_sum = Nx.sum(weights, axes: [0], keep_axes: true)

          weights =
            Nx.select(Nx.abs(weights) > 1000 * @f32_eps, safe_divide(weights, weights_sum), 0)

          input = Nx.dot(input, [axis], weights, [0])
          # The transformed axis is moved to the end, so we transpose back
          reorder_axis(input, -1, axis)
      end
    end)
  end

  defnp fill_linear_kernel(x) do
    Nx.max(0, 1 - x)
  end

  defnp fill_cubic_kernel(x) do
    # See https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    out = (1.5 * x - 2.5) * x * x + 1
    out = Nx.select(x >= 1, ((-0.5 * x + 2.5) * x - 4) * x + 2, out)
    Nx.select(x >= 2, 0, out)
  end

  @pi :math.pi()

  defnp fill_lanczos_kernel(radius, x) do
    y = radius * Nx.sin(@pi * x) * Nx.sin(@pi * x / radius)
    out = Nx.select(x > 1.0e-3, safe_divide(y, @pi ** 2 * x ** 2), 1)
    Nx.select(x > radius, 0, out)
  end

  defnp safe_divide(x, y) do
    x / Nx.select(y != 0, y, 1)
  end

  defnp reorder_axis(tensor, axis, target_axis) do
    transform({tensor, axis, target_axis}, fn {tensor, axis, target_axis} ->
      axes = Nx.axes(tensor)
      {source_axis, axes} = List.pop_at(axes, axis)
      axes = List.insert_at(axes, target_axis, source_axis)
      Nx.transpose(tensor, axes: axes)
    end)
  end

  @doc """
  Scales an image such that the short edge matches the given size.

  ## Options

    * `:size` - the target size of the short edge

    * `:method` - the resizing method to use, same as `resize/2`

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  """
  defn resize_short(input, opts \\ []) do
    opts = keyword!(opts, [:size, channels: :last, method: :bilinear])

    size = opts[:size]
    method = opts[:method]
    channels = opts[:channels]

    {height, width} = size(input, channels: channels)
    {out_height, out_width} = transform({height, width, size}, &resize_short_size/1)

    resize(input, size: {out_height, out_width}, method: method, channels: channels)
  end

  defp resize_short_size({height, width, size}) do
    {short, long} = if height < width, do: {height, width}, else: {width, height}

    out_short = size
    out_long = floor(size * long / short)

    if height < width, do: {out_short, out_long}, else: {out_long, out_short}
  end

  @doc """
  Returns the image size as `{height, width}`.

  ## Options

      * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  """
  defn size(input, opts \\ []) do
    opts = keyword!(opts, channels: :last)
    {height_axis, width_axis} = spatial_axes(input, channels: opts[:channels])
    {Nx.axis_size(input, height_axis), Nx.axis_size(input, width_axis)}
  end

  @doc """
  Normalizes an image according to the given per-channel mean and
  standard deviation.

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  """
  defn normalize(input, mean, std, opts \\ []) do
    opts = keyword!(opts, channels: :last)

    mean = broadcast_channel_info(mean, input, opts[:channels])
    std = broadcast_channel_info(std, input, opts[:channels])

    (input - mean) / std
  end

  deftransformp broadcast_channel_info(tensor, input, channels) do
    rank = Nx.rank(input)
    type = Nx.type(input)

    channels_axis =
      case channels do
        :first -> rank - 3
        :last -> rank - 1
      end

    shape = 1 |> Tuple.duplicate(rank) |> put_elem(channels_axis, :auto)

    tensor |> Nx.reshape(shape) |> Nx.as_type(type)
  end

  @doc """
  Normalizes an image size to a `{height, width}` tuple.

  Accepts either an existing tuple or a single number used for both
  dimensions.
  """
  def normalize_size(size)

  def normalize_size({height, width}), do: {height, width}
  def normalize_size(size) when is_integer(size), do: {size, size}

  @doc """
  Converts pixel values (0-255) into a continuous range.
  """
  defn to_continuous(input, min, max) do
    input / 255.0 * (max - min) + min
  end

  @doc """
  Converts values from continuous range into pixel values (0-255).
  """
  defn from_continuous(input, min, max) do
    input = (input - min) / (max - min) * 255.0

    input
    |> Nx.round()
    |> Nx.clip(0, 255)
    |> Nx.as_type(:u8)
  end
end
