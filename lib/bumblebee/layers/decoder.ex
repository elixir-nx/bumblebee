defmodule Bumblebee.Layers.Decoder do
  @moduledoc false

  import Nx.Defn

  alias Bumblebee.Layers

  @doc """
  Builds a fresh cache for iterative decoding.

  ## Caching

  Sequence generation involves iterative inference using the decoder
  prat of the model. The inference is autoregressive, which means that
  the decoder output token is appended to the input sequence for the
  next iteration. Passing all tokens on every iteration is wasteful,
  because certain hidden states are always the same for the same token,
  so they would be unnecessarily recomputed on every subsequent iteration.
  To reduce the complexity, we can feed a single input token at a time,
  put those hidden states in a cache and use on subsequent iterations.

  For self-attention blocks, we cache the key and value state for the
  input token and append them to the cache.

  For cross-attention blocks, we compute the whole key and value state
  on the first iteration and reuse on subsequent ones.

  We also accumulate an attention mask corresponding to each token.

  In order to reuse compiled computation, we want the cache to keep
  the same shape, so we initialize it for a fixed max length of the
  decoder sequence.

  ## Options

    * `:hidden_size` - the dimensionality of the hidden layers

    * `:attention_head_size` - the size of the key, value, and query
      projection per attention head

    * `:key_head_size` - the size of the key projection per attention
      head, when it differs from `:attention_head_size`

    * `:value_head_size` - the size of the value projection per attention
      head, when it differs from `:attention_head_size`

    * `:extra_states` - additional per-position states to cache in every
      self-attention block. Expects a keyword list with the state name
      and its shape, excluding the leading batch size and sequence
      length. See `cached_state/4`

    * `:extra_window_states` - additional fixed-size window states to
      cache in every self-attention block. Expects a keyword list with
      the state name and its shape, excluding the leading batch size.
      See `cached_window_state/3`

    * `:decoder_num_blocks` - the number of Transformer blocks in the decoder

    * `:decoder_num_attention_heads` - the number of decoder attention heads

  Note that `:attention_head_size`, `:key_head_size`, `:value_head_size`,
  `:decoder_num_attention_heads` and `:extra_states` may also be given as
  functions of the block index, for models where the attention shape
  differs across blocks.

    * `:encoder_num_attention_heads` - the number of encoder attention heads
      (for cross attention)

    * `:encoder_sequence_length` - the encoder input sequence length
      (for cross attention)

  """
  def init_cache(batch_size, max_length, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    decoder_num_attention_heads = Keyword.fetch!(opts, :decoder_num_attention_heads)
    decoder_num_blocks = Keyword.fetch!(opts, :decoder_num_blocks)
    encoder_num_attention_heads = opts[:encoder_num_attention_heads]
    encoder_sequence_length = opts[:encoder_sequence_length]

    self_attention = fn block_idx ->
      num_attention_heads = per_block(decoder_num_attention_heads, block_idx)

      decoder_head_size =
        per_block(opts[:attention_head_size], block_idx) ||
          div(hidden_size, num_attention_heads)

      self_attention =
        attention_cache(
          batch_size,
          max_length,
          num_attention_heads,
          per_block(opts[:key_head_size], block_idx) || decoder_head_size,
          per_block(opts[:value_head_size], block_idx) || decoder_head_size
        )

      self_attention =
        for {name, shape} <- per_block(opts[:extra_states], block_idx) || [],
            into: self_attention do
          {name,
           Nx.broadcast(0.0, List.to_tuple([batch_size, max_length | Tuple.to_list(shape)]))}
        end

      for {name, shape} <- per_block(opts[:extra_window_states], block_idx) || [],
          into: self_attention do
        {name, Nx.broadcast(0.0, List.to_tuple([batch_size | Tuple.to_list(shape)]))}
      end
    end

    cross_attention =
      if encoder_sequence_length do
        encoder_head_size =
          opts[:attention_head_size] || div(hidden_size, encoder_num_attention_heads)

        attention_cache(
          batch_size,
          encoder_sequence_length,
          encoder_num_attention_heads,
          encoder_head_size,
          encoder_head_size
        )
      else
        %Axon.None{}
      end

    blocks =
      for block_idx <- 0..(decoder_num_blocks - 1) do
        %{self_attention: self_attention.(block_idx), cross_attention: cross_attention}
      end
      |> List.to_tuple()

    offset = Nx.tensor(0)

    attention_mask = Nx.broadcast(0, {batch_size, max_length})

    %{blocks: blocks, offset: offset, attention_mask: attention_mask}
  end

  defp per_block(fun, block_idx) when is_function(fun, 1), do: fun.(block_idx)
  defp per_block(value, _block_idx), do: value

  defp attention_cache(batch_size, sequence_length, num_heads, key_head_size, value_head_size) do
    key = Nx.broadcast(0.0, {batch_size, sequence_length, num_heads, key_head_size})
    value = Nx.broadcast(0.0, {batch_size, sequence_length, num_heads, value_head_size})
    %{key: key, value: value}
  end

  @doc """
  Calls `fun` for every batched tensor in cache initialized with
  `init_cache/3`.
  """
  def traverse_cache(cache, fun) do
    %{blocks: blocks, offset: offset, attention_mask: attention_mask} = cache

    %{
      blocks: Bumblebee.Utils.Nx.map(blocks, fun),
      offset: offset,
      attention_mask: fun.(attention_mask)
    }
  end

  @doc """
  Combines new attention mask with the one in cache.

  The function returns the full attention mask and the updated cache.
  """
  def cached_attention_mask(attention_mask, cache) do
    Layers.if_present cache do
      Axon.layer(
        fn attention_mask, cache, _ ->
          indices = [0, Nx.as_type(cache.offset, {:s, 64})]
          attention_mask = Nx.put_slice(cache.attention_mask, indices, attention_mask)
          {attention_mask, %{cache | attention_mask: attention_mask}}
        end,
        [attention_mask, cache]
      )
      |> Layers.unwrap_tuple(2)
    else
      {attention_mask, cache}
    end
  end

  @doc """
  Combines new key and value state with those in cache.

  The function returns the full key and value state and the updated
  cache.
  """
  def cached_attention_key_values(key, value, attention_cache, offset) do
    Layers.if_present attention_cache do
      Axon.layer(&update_attention_cache/5, [key, value, attention_cache, offset])
      |> Layers.unwrap_tuple(3)
    else
      {key, value, attention_cache}
    end
  end

  deftransformp update_attention_cache(key, value, attention_cache, offset, _opts \\ []) do
    if Nx.shape(key) == Nx.shape(attention_cache.key) and
         Nx.shape(value) == Nx.shape(attention_cache.value) do
      # If the shapes are the same, it means the keys and values would
      # be the same on every iteration, so we take cache all of them at
      # at once (as in cross-attention)
      replace_attention_cache(key, value, attention_cache, offset)
    else
      # Otherwise we compute keys and values incrementally, so we append
      # to the cache (as in self-attention)
      append_attention_cache(key, value, attention_cache, offset)
    end
  end

  defnp replace_attention_cache(key, value, attention_cache, offset, _opts \\ []) do
    if offset == 0 do
      attention_cache = %{attention_cache | key: key, value: value}
      {key, value, attention_cache}
    else
      {attention_cache.key, attention_cache.value, attention_cache}
    end
  end

  defnp append_attention_cache(key, value, attention_cache, offset, _opts \\ []) do
    %{key: cached_key, value: cached_value} = attention_cache
    indices = [0, offset, 0, 0]
    key = Nx.put_slice(cached_key, indices, key)
    value = Nx.put_slice(cached_value, indices, value)
    updated_cache = %{attention_cache | key: key, value: value}
    {key, value, updated_cache}
  end

  @doc """
  Combines a new auxiliary state with the one in cache.

  This is the equivalent of `cached_attention_key_values/4` for extra
  per-position states, such as the keys of the sparse attention indexer
  in the DeepSeek V3.2 family. The state is accumulated along the second
  axis and the corresponding cache entry must be initialized by passing
  `:extra_states` to `init_cache/3`.
  """
  def cached_state(state, name, attention_cache, offset) do
    Layers.if_present attention_cache do
      Axon.layer(&update_state_cache/4, [state, attention_cache, offset], state_name: name)
      |> Layers.unwrap_tuple(2)
    else
      {state, attention_cache}
    end
  end

  deftransformp update_state_cache(state, attention_cache, offset, opts \\ []) do
    name = opts[:state_name]
    cached_state = Map.fetch!(attention_cache, name)

    state =
      if Nx.shape(state) == Nx.shape(cached_state) do
        state
      else
        indices = [0, offset] ++ List.duplicate(0, Nx.rank(cached_state) - 2)
        Nx.put_slice(cached_state, indices, state)
      end

    {state, Map.put(attention_cache, name, state)}
  end

  @doc """
  Reads an extra state from the cache.

  Returns `Bumblebee.Layers.none/0` when there is no cache. The
  corresponding cache entry must be initialized by passing
  `:extra_window_states` to `init_cache/3`.
  """
  def get_extra_state(attention_cache, name) do
    Layers.if_present attention_cache do
      Axon.layer(
        fn attention_cache, opts -> Map.fetch!(attention_cache, opts[:state_name]) end,
        [attention_cache],
        state_name: name,
        op_name: :get_extra_state
      )
    else
      Layers.none()
    end
  end

  @doc """
  Writes an extra state into the cache.
  """
  def put_extra_state(attention_cache, name, state) do
    Layers.if_present attention_cache do
      Axon.layer(
        fn attention_cache, state, opts ->
          Map.put(attention_cache, opts[:state_name], state)
        end,
        [attention_cache, state],
        state_name: name,
        op_name: :put_extra_state
      )
    else
      attention_cache
    end
  end

  @doc """
  Combines a new state with a fixed-size window of preceding states from
  the cache.

  Returns the concatenation of the cached window and the new state, as
  well as the updated cache, where the window is the trailing part of the
  concatenation. This is used by models with short convolutions, which
  only need a few preceding positions. The corresponding cache entry must
  be initialized by passing `:extra_states` to `init_cache/3`.
  """
  def cached_window_state(state, name, attention_cache) do
    Layers.if_present attention_cache do
      Axon.layer(&update_window_cache/3, [state, attention_cache], state_name: name)
      |> Layers.unwrap_tuple(2)
    else
      {state, attention_cache}
    end
  end

  deftransformp update_window_cache(state, attention_cache, opts \\ []) do
    name = opts[:state_name]
    window = Map.fetch!(attention_cache, name)
    window_size = Nx.axis_size(window, 1)

    full_state = Nx.concatenate([window, state], axis: 1)

    window =
      Nx.slice_along_axis(
        full_state,
        Nx.axis_size(full_state, 1) - window_size,
        window_size,
        axis: 1
      )

    {full_state, Map.put(attention_cache, name, window)}
  end

  @doc """
  Retrieves cache for a specific decoder block.
  """
  def get_block_cache(cache, block_idx) do
    Axon.nx(cache, &elem(&1.blocks, block_idx))
  end

  @doc """
  Puts an updated cache entry for the given decoder block.
  """
  def put_block_cache(cache, block_idx, block_cache) do
    Axon.layer(
      fn cache, block_cache, _opts ->
        put_in(cache, [:blocks, Access.elem(block_idx)], block_cache)
      end,
      [cache, block_cache]
    )
  end

  @doc """
  Retrieves self-attention and cross-attention caches from a block
  cache.
  """
  def get_attention_caches(block_cache) do
    {Axon.nx(block_cache, & &1.self_attention), Axon.nx(block_cache, & &1.cross_attention)}
  end

  @doc """
  Puts updated self-attention and cross-attention cache entries for
  in the decoder block cache.
  """
  def put_attention_caches(block_cache, self_attention_cache, cross_attention_cache) do
    Axon.layer(
      fn block_cache, self_attention_cache, cross_attention_cache, _opts ->
        %{
          block_cache
          | self_attention: self_attention_cache,
            cross_attention: cross_attention_cache
        }
      end,
      [block_cache, Axon.optional(self_attention_cache), Axon.optional(cross_attention_cache)]
    )
  end

  @doc """
  Retrieves the cache offset.

  Cache offset keeps track of how many input tokens were processed
  and are in the cache.
  """
  def get_cache_offset(cache) do
    Axon.nx(cache, & &1.offset)
  end

  @doc """
  Bumps the cache offset by the number of tokens in `input_embeddings`.
  """
  def update_cache_offset(cache, input_embeddings) do
    Axon.layer(
      fn cache, input_embeddings, _ ->
        sequence_length = Nx.axis_size(input_embeddings, 1)
        update_in(cache.offset, &Nx.add(&1, sequence_length))
      end,
      [cache, input_embeddings]
    )
  end
end
