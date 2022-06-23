defmodule Bumblebee.Utils.Image do
  @moduledoc false

  import Nx.Defn

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
    |> Nx.transpose(axes: [:channels, :height, :width])
    |> Nx.new_axis(0, :batch)
  end

  @doc """
  Crops images at the center.

  If the image is too small to be cropped to the desired size, it gets
  padded with zeros.

  ## Options

    * `:size` - the target image size specified as `{height, width}`

  ## Examples

      iex> images = Nx.iota({1, 1, 4, 4})
      iex> Bumblebee.Utils.Image.center_crop(images, size: {2, 2})
      #Nx.Tensor<
        s64[1][1][2][2]
        [
          [
            [
              [5, 6],
              [9, 10]
            ]
          ]
        ]
      >

      iex> images = Nx.iota({1, 1, 2, 2})
      iex> Bumblebee.Utils.Image.center_crop(images, size: {1, 4})
      #Nx.Tensor<
        s64[1][1][1][4]
        [
          [
            [
              [0, 0, 1, 0]
            ]
          ]
        ]
      >

  """
  defn center_crop(images, opts \\ []) do
    opts = keyword!(opts, [:size])

    pad_config =
      transform({images, opts[:size]}, fn {images, {out_height, out_width}} ->
        {height, width} = size(images)
        top = div(height - out_height, 2)
        bottom = top + out_height
        left = div(width - out_width, 2)
        right = left + out_width

        [{0, 0, 0}, {0, 0, 0}, {-top, bottom - height, 0}, {-left, right - width, 0}]
      end)

    Nx.pad(images, 0, pad_config)
  end

  @doc """
  Resizes the images.

  ## Options

    * `:size` - the target image size specified as `{height, width}`

    * `:method` - the resizing method to use, either of `:nearest`,
      `:linear`, `:cubic`, `:lanczos3`, `:lanczos5`. Defaults to
      `:linear`

  ## Examples

      iex> images = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      iex> Bumblebee.Utils.Image.resize(images, size: {3, 3}, method: :nearest)
      #Nx.Tensor<
        f32[1][1][3][3]
        [
          [
            [
              [0.0, 1.0, 1.0],
              [2.0, 3.0, 3.0],
              [2.0, 3.0, 3.0]
            ]
          ]
        ]
      >
      iex> Bumblebee.Utils.Image.resize(images, size: {3, 3}, method: :linear)
      #Nx.Tensor<
        f32[1][1][3][3]
        [
          [
            [
              [0.0, 0.5, 1.0],
              [1.0, 1.5, 2.0],
              [2.0, 2.5, 3.0]
            ]
          ]
        ]
      >

  """
  defn resize(images, opts \\ []) do
    opts = keyword!(opts, [:size, method: :linear])

    transform({images, opts[:size], opts[:method]}, fn {images, size, method} ->
      {batch, channels, height, width} = Nx.shape(images)
      {out_height, out_width} = size

      output_shape = {batch, channels, out_height, out_width}

      spacial_axes =
        for {axis, size, out_size} <- [{2, height, out_height}, {3, width, out_width}],
            Elixir.Kernel.!=(size, out_size),
            do: axis

      resized_images =
        case method do
          :nearest ->
            resize_nearest(images, output_shape, spacial_axes)

          :linear ->
            resize_with_kernel(images, output_shape, spacial_axes, &fill_linear_kernel/1)

          :cubic ->
            resize_with_kernel(images, output_shape, spacial_axes, &fill_cubic_kernel/1)

          :lanczos3 ->
            resize_with_kernel(images, output_shape, spacial_axes, &fill_lanczos_kernel(3, &1))

          :lanczos5 ->
            resize_with_kernel(images, output_shape, spacial_axes, &fill_lanczos_kernel(5, &1))

          method ->
            raise ArgumentError,
                  "expected :method to be either of :nearest, :linear, :cubic, " <>
                    ":lanczos3, :lanczos5, got: #{inspect(method)}"
        end

      cast_to(resized_images, images)
    end)
  end

  defnp cast_to(left, right) do
    left
    |> Nx.as_type(Nx.type(right))
    |> Nx.reshape(left, names: Nx.names(right))
  end

  defnp resize_nearest(images, output_shape, spacial_axes) do
    transform({images, output_shape, spacial_axes}, fn {images, output_shape, spacial_axes} ->
      singular_shape = List.duplicate(1, Nx.rank(images)) |> List.to_tuple()

      for axis <- spacial_axes, reduce: images do
        images ->
          input_shape = Nx.shape(images)
          input_size = elem(input_shape, axis)
          output_size = elem(output_shape, axis)
          inv_scale = input_size / output_size
          offset = (Nx.iota({output_size}) + 0.5) * inv_scale
          offset = offset |> Nx.floor() |> Nx.as_type({:s, 32})

          offset =
            offset
            |> Nx.reshape(put_elem(singular_shape, axis, output_size))
            |> Nx.broadcast(put_elem(input_shape, axis, output_size))

          Nx.take_along_axis(images, offset, axis: axis)
      end
    end)
  end

  @f32_eps :math.pow(2, -23)

  defnp resize_with_kernel(images, output_shape, spacial_axes, kernel_fun) do
    transform({images, output_shape, spacial_axes}, fn {images, output_shape, spacial_axes} ->
      for axis <- spacial_axes, reduce: images do
        images ->
          input_shape = Nx.shape(images)
          input_size = elem(input_shape, axis)
          output_size = elem(output_shape, axis)

          inv_scale = input_size / output_size
          kernel_scale = Nx.max(1, inv_scale)

          sample_f = (Nx.iota({1, output_size}) + 0.5) * inv_scale - 0.5
          x = Nx.abs(sample_f - Nx.iota({input_size, 1})) / kernel_scale
          weights = kernel_fun.(x)

          weights_sum = Nx.sum(weights, axes: [0], keep_axes: true)

          weights =
            Nx.select(Nx.abs(weights) > 1000 * @f32_eps, safe_divide(weights, weights_sum), 0)

          images = Nx.dot(images, [axis], weights, [0])
          # The transformed axis is moved to the end, so we transpose back
          Nx.transpose(images, axes: List.insert_at([0, 1, 2], axis, 3))
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

  @doc """
  Scales images such that the short edge matches the given size.

  ## Options

    * `:size` - the target size of the short edge

    * `:method` - the resizing method to use, same as `resize/2`

  """
  defn resize_short(images, opts \\ []) do
    opts = keyword!(opts, [:size, method: :linear])

    size = opts[:size]
    method = opts[:method]

    {height, width} = size(images)
    {out_height, out_width} = transform({height, width, size}, &resize_short_size/1)

    resize(images, size: {out_height, out_width}, method: method)
  end

  defp resize_short_size({height, width, size}) do
    {short, long} = if height < width, do: {height, width}, else: {width, height}

    out_short = size
    out_long = floor(size * long / short)

    if height < width, do: {out_short, out_long}, else: {out_long, out_short}
  end

  @doc """
  Returns the image size as `{height, width}`.
  """
  defn size(images) do
    {_batch, _channels, height, width} = Nx.shape(images)
    {height, width}
  end
end
