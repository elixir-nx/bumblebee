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

    * `:decoder_num_blocks` - the number of Transformer blocks in the decoder

    * `:decoder_num_attention_heads` - the number of decoder attention heads

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

    self_attention =
      attention_cache(batch_size, max_length, hidden_size, decoder_num_attention_heads)

    cross_attention =
      if encoder_sequence_length do
        attention_cache(
          batch_size,
          encoder_sequence_length,
          hidden_size,
          encoder_num_attention_heads
        )
      else
        %Axon.None{}
      end

    blocks =
      %{self_attention: self_attention, cross_attention: cross_attention}
      |> List.duplicate(decoder_num_blocks)
      |> List.to_tuple()

    offset = Nx.tensor(0.0)

    attention_mask = Nx.broadcast(0.0, {batch_size, max_length})

    %{blocks: blocks, offset: offset, attention_mask: attention_mask}
  end

  defp attention_cache(batch_size, sequence_length, hidden_size, num_heads) do
    head_size = div(hidden_size, num_heads)
    shape = {batch_size, sequence_length, num_heads, head_size}
    zeros = Nx.broadcast(0.0, shape)
    %{key: zeros, value: zeros}
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
  def cached_attention_key_values(key, value, attention_cache, offset, opts \\ []) do
    opts = Keyword.validate!(opts, cross_attention?: false)

    update_fun =
      if opts[:cross_attention?],
        do: &update_cross_attention_cache/5,
        else: &update_self_attention_cache/5

    Layers.if_present attention_cache do
      Axon.layer(update_fun, [key, value, attention_cache, offset])
      |> Layers.unwrap_tuple(3)
    else
      {key, value, attention_cache}
    end
  end

  defnp update_self_attention_cache(key, value, attention_cache, offset, _opts \\ []) do
    %{key: cached_key, value: cached_value} = attention_cache
    indices = [0, Nx.as_type(offset, {:s, 64}), 0, 0]
    key = Nx.put_slice(cached_key, indices, key)
    value = Nx.put_slice(cached_value, indices, value)
    updated_cache = %{key: key, value: value}
    {key, value, updated_cache}
  end

  defnp update_cross_attention_cache(key, value, attention_cache, offset, _opts \\ []) do
    if offset == 0 do
      attention_cache = %{attention_cache | key: key, value: value}
      {key, value, attention_cache}
    else
      {attention_cache.key, attention_cache.value, attention_cache}
    end
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

  @doc """
  Builds a causal mask and combines it with the given attention mask.

  A causal mask is used to mask bidirectional self-attention, such
  that it works in a single direction.

  Accepts an optional offset, which should be set when passing a
  partial query.
  """
  def apply_causal_mask(attention_mask, query, offset) do
    Axon.layer(
      fn attention_mask, query, offset, _opts ->
        sequence_length = Nx.axis_size(attention_mask, -1)

        # We generate a full causal mask, then slice it in case of
        # iterative decoding
        causal_mask = build_causal_mask(Nx.broadcast(1, {1, sequence_length}))

        causal_mask =
          case offset do
            %Axon.None{} ->
              causal_mask

            offset ->
              mask_shift = Nx.as_type(offset, {:s, 64})
              query_length = Nx.axis_size(query, 1)
              Nx.slice_along_axis(causal_mask, mask_shift, query_length, axis: 2)
          end

        Nx.logical_and(attention_mask, causal_mask)
      end,
      [attention_mask, query, Axon.optional(offset)]
    )
  end

  defnp build_causal_mask(input) do
    size = Nx.axis_size(input, -1)
    idx = Nx.iota({size}) |> Nx.broadcast(input)
    build_attention_mask(idx, idx)
  end

  # Expects a batched, flat inputs of length corresponding to query
  # and key length respectively.
  defnp build_attention_mask(query_input, key_input) do
    query_input
    |> Nx.new_axis(-1)
    |> Nx.greater_equal(Nx.new_axis(key_input, -2))
    |> Nx.new_axis(-3)
  end
end
