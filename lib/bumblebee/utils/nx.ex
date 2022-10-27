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
    {batch_size, axis_size, flat_shape, final_shape} =
      transform({tensor, idx}, fn {tensor, idx} ->
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
      end)

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
  A variant of `Nx.to_batched/2` which also works on maps.
  """
  def to_batched(tensor_or_container, batch_size) do
    case tensor_or_container do
      %Nx.Tensor{} = tensor ->
        Nx.to_batched(tensor, batch_size)

      container ->
        container
        |> Enum.map(fn {key, batch} ->
          list_of_tensors = batch |> Nx.to_batched(1) |> Enum.to_list()
          Enum.map(list_of_tensors, fn tensor -> {key, tensor} end)
        end)
        |> Enum.zip_with(&Map.new/1)
    end
  end

  @doc """
  Computes cosine similarity between the given tensors.
  """
  defn cosine_similarity(x, y) do
    x = normalize(x)
    y = normalize(y)
    Nx.dot(x, [-1], y, [-1])
  end

  defnp normalize(tensor) do
    norm =
      tensor
      |> Nx.power(2)
      |> Nx.sum(axes: [-1], keep_axes: true)
      |> Nx.sqrt()

    tensor / norm
  end
end
