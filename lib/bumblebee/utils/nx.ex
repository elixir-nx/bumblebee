defmodule Bumblebee.Utils.Nx do
  @moduledoc false

  import Nx.Defn

  @doc """
  Recursively maps the given container with a function.

  When a tensor is given, the function is applied to this tensor.
  """
  @spec map(Nx.Tensor.t() | Nx.Container.t(), (Nx.Tensor.t() -> term())) :: term()
  def map(container_or_tensor, fun)

  def map(%Nx.Tensor{} = tensor, fun) do
    fun.(tensor)
  end

  def map(container, fun) do
    container
    |> Nx.Container.traverse(nil, fn item, acc ->
      {map(item, fun), acc}
    end)
    |> elem(0)
  end

  @doc """
  Returns the underlying tensor as a list.

  The list nesting matches the tensor shape.
  """
  @spec to_list(Nx.Tensor.t()) :: list()
  def to_list(tensor) do
    list = Nx.to_flat_list(tensor)

    tensor
    |> Nx.shape()
    |> Tuple.to_list()
    |> Enum.drop(1)
    |> Enum.reverse()
    |> Enum.reduce(list, &Stream.chunk_every(&2, &1))
    |> Enum.to_list()
  end

  @doc """
  Splits tensor or container along the first axis.

  Note: this function traverses the container N times, where N is the
  batch size. Don't use it for containers with a lot of tensors.

  ## Examples

      iex> outputs = %{x: Nx.tensor([[0, 0], [1, 1]]), y: Nx.tensor([0, 1])}
      iex> [first, second] = Bumblebee.Utils.Nx.batch_to_list(outputs)
      iex> first.x
      #Nx.Tensor<
        s64[2]
        [0, 0]
      >
      iex> second.x
      #Nx.Tensor<
        s64[2]
        [1, 1]
      >
      iex> first.y
      #Nx.Tensor<
        s64
        0
      >
      iex> second.y
      #Nx.Tensor<
        s64
        1
      >

  """
  @spec batch_to_list(Nx.Tensor.t() | Nx.Container.t()) :: list(Nx.Tensor.t() | Nx.Container.t())
  def batch_to_list(tensor_or_container) do
    batch_size =
      Nx.Defn.Composite.reduce(tensor_or_container, nil, fn
        tensor, nil -> Nx.axis_size(tensor, 0)
        tensor, size -> ^size = Nx.axis_size(tensor, 0)
      end)

    for idx <- 0..(batch_size - 1)//1 do
      Nx.Defn.Composite.traverse(tensor_or_container, fn tensor -> tensor[idx] end)
    end
  end

  @doc """
  Concatenates corresponding tensors in matching containers.

  ## Options

    * `:axis` - the axis to concatenate along. Defaults to `0`

  ## Examples

      iex> left = %{x: Nx.tensor([[0, 0], [1, 1]]), y: Nx.tensor([0, 1])}
      iex> right = %{x: Nx.tensor([[2, 2], [3, 3]]), y: Nx.tensor([2, 3])}
      iex> result = Bumblebee.Utils.Nx.composite_concatenate(left, right)
      iex> result.x
      #Nx.Tensor<
        s64[4][2]
        [
          [0, 0],
          [1, 1],
          [2, 2],
          [3, 3]
        ]
      >
      iex> result.y
      #Nx.Tensor<
        s64[4]
        [0, 1, 2, 3]
      >

  """
  deftransform composite_concatenate(left, right, opts \\ []) do
    opts = Keyword.validate!(opts, axis: 0)

    axis = opts[:axis]

    right_tensors =
      right
      |> Nx.Defn.Composite.reduce([], &[&1 | &2])
      |> Enum.reverse()

    {result, []} =
      Nx.Defn.Composite.traverse(left, right_tensors, fn left, [right | right_tensors] ->
        {Nx.concatenate([left, right], axis: axis), right_tensors}
      end)

    result
  end

  @doc """
  Reshapes the tensor to have a new leading axis of the given size.

  ## Examples

      iex> output = %{x: Nx.tensor([[0, 0], [1, 1]]), y: Nx.tensor([0, 1])}
      iex> result = Bumblebee.Utils.Nx.composite_unflatten_batch(output, 2)
      iex> result.x
      #Nx.Tensor<
        s64[2][1][2]
        [
          [
            [0, 0]
          ],
          [
            [1, 1]
          ]
        ]
      >
      iex> result.y
      #Nx.Tensor<
        s64[2][1]
        [
          [0],
          [1]
        ]
      >

  """
  deftransform composite_unflatten_batch(container, batch_size) do
    map(container, fn tensor ->
      shape =
        tensor
        |> Nx.shape()
        |> Tuple.insert_at(0, batch_size)
        |> put_elem(1, :auto)

      Nx.reshape(tensor, shape)
    end)
  end

  @doc """
  Flattens two leading tensor axes into a single axis.

  ## Examples

      iex> output = %{x: Nx.tensor([[0, 0], [1, 1]]), y: Nx.tensor([[0], [1]])}
      iex> result = Bumblebee.Utils.Nx.composite_flatten_batch(output)
      iex> result.x
      #Nx.Tensor<
        s64[4]
        [0, 0, 1, 1]
      >
      iex> result.y
      #Nx.Tensor<
        s64[2]
        [0, 1]
      >

  """
  deftransform composite_flatten_batch(container) do
    map(container, fn tensor ->
      shape =
        tensor
        |> Nx.shape()
        |> Tuple.delete_at(0)
        |> put_elem(0, :auto)

      Nx.reshape(tensor, shape)
    end)
  end

  @doc """
  A version of `Nx.take/3` with a leading batch dimension.

  Conceptually, this function zips `tensor` with `indices` and then
  applies `Nx.take/3` along the first axis for every pair.

  ## Examples

      iex> t =
      ...>   Nx.tensor([
      ...>     [
      ...>       [1, 1],
      ...>       [2, 2]
      ...>     ],
      ...>     [
      ...>       [3, 3],
      ...>       [4, 4]
      ...>     ]
      ...>   ])
      iex> idx = Nx.tensor([[1, 0], [1, 1]])
      iex> Bumblebee.Utils.Nx.batched_take(t, idx)
      #Nx.Tensor<
        s64[2][2][2]
        [
          [
            [2, 2],
            [1, 1]
          ],
          [
            [4, 4],
            [4, 4]
          ]
        ]
      >

  ## Error cases

      iex> Bumblebee.Utils.Nx.batched_take(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1]))
      ** (ArgumentError) expected tensor and indices with the same leading axis size, got: {2, 2} and {1}

  """
  defn batched_take(tensor, idx) do
    {batch_size, axis_size, flat_shape, final_shape} = get_batched_take_shapes(tensor, idx)

    flat_idx =
      idx
      |> Nx.reshape({batch_size, :auto})
      |> Nx.add(Nx.iota({batch_size, 1}) * axis_size)
      |> Nx.flatten()

    tensor
    |> Nx.reshape(flat_shape)
    |> Nx.take(flat_idx)
    |> Nx.reshape(final_shape)
  end

  deftransformp get_batched_take_shapes(tensor, idx) do
    tensor_shape = Nx.shape(tensor)
    idx_shape = Nx.shape(idx)

    unless Elixir.Kernel.==(elem(tensor_shape, 0), elem(idx_shape, 0)) do
      raise ArgumentError,
            "expected tensor and indices with the same leading axis size, got: #{inspect(tensor_shape)} and #{inspect(idx_shape)}"
    end

    [batch_size, axis_size | inner_sizes] = Tuple.to_list(tensor_shape)

    flat_shape = List.to_tuple([batch_size * axis_size | inner_sizes])
    final_shape = List.to_tuple(Tuple.to_list(idx_shape) ++ inner_sizes)
    {batch_size, axis_size, flat_shape, final_shape}
  end

  @doc """
  Creates a tensor with evenly spaced elements within the given range.

  The elements are taken from the interval `[start, stop]`, both ends
  are included.

  ## Options

    * `:steps` - the number of samples in the range. Defaults to `50`

  ## Examples

      iex> Bumblebee.Utils.Nx.linspace(2.0, 3.0, steps: 5)
      #Nx.Tensor<
        f32[5]
        [2.0, 2.25, 2.5, 2.75, 3.0]
      >

  """
  defn linspace(start, stop, opts \\ []) do
    opts = keyword!(opts, steps: 50)
    steps = opts[:steps]

    step_size = (stop - start) / (steps - 1)
    Nx.iota({steps}) * step_size + start
  end

  @doc """
  Computes cosine similarity between the given tensors.

  ## Options

    * `:batched?` - if `true`, treats inputs as batches along the first
      axis and computes individual similarities. Defaults to `false`

  """
  deftransform cosine_similarity(x, y, opts \\ []) do
    opts = Keyword.validate!(opts, batched?: false)
    batched? = opts[:batched?]

    x = normalize(x)
    y = normalize(y)

    batch_axes = if batched?, do: [0], else: []

    Nx.dot(x, [-1], batch_axes, y, [-1], batch_axes)
  end

  defn normalize(tensor) do
    norm =
      tensor
      |> Nx.pow(2)
      |> Nx.sum(axes: [-1], keep_axes: true)
      |> Nx.sqrt()

    norm = Nx.select(norm == 0.0, 1.0, norm)
    
    tensor / norm
  end

  @doc """
  Repeats tensor along the given axis, such that repeated chunks are
  adjacent to each other.

  ## Options

    * `:axis` - the axis to repeat along. Defaults to `0`

  ## Examples

    iex> x = Nx.tensor([[1, 2], [3, 4]])
    iex> Bumblebee.Utils.Nx.repeat_interleave(x, 2)
    #Nx.Tensor<
      s64[4][2]
      [
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4]
      ]
    >

  """
  deftransform repeat_interleave(tensor, times, opts \\ []) do
    opts = Keyword.validate!(opts, axis: 0)

    axis = opts[:axis]

    repetitions =
      1
      |> List.duplicate(Nx.rank(tensor))
      |> List.insert_at(axis + 1, times)

    tensor
    |> Nx.new_axis(axis + 1)
    |> Nx.tile(repetitions)
    |> Nx.flatten(axes: [axis, axis + 1])
  end

  @doc """
  A version of `Nx.take/3` with each index applying to a corresponding
  chunk.

  ## Options

    * `:axis` - the axis to take along. Defaults to `0`

  ## Examples

    iex> x = Nx.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    iex> Bumblebee.Utils.Nx.chunked_take(x, 2, Nx.tensor([1, 0]))
    #Nx.Tensor<
      s64[2][2]
      [
        [2, 2],
        [3, 3]
      ]
    >

  """
  deftransform chunked_take(tensor, chunk_size, idx, opts \\ []) do
    opts = Keyword.validate!(opts, axis: 0)

    flat_idx =
      idx
      |> Nx.shape()
      |> Nx.iota()
      |> Nx.multiply(chunk_size)
      |> Nx.add(idx)

    Nx.take(tensor, flat_idx, axis: opts[:axis])
  end
end
