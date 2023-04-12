defmodule Bumblebee.Text.Generation.LogitsProcessing do
  @moduledoc false

  import Nx.Defn

  defn bos_token_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:bos_token_id])
    bos_token_id = opts[:bos_token_id]

    if context.length == 1 do
      force_token_id(logits, bos_token_id)
    else
      logits
    end
  end

  defn eos_token_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:eos_token_id])
    eos_token_id = opts[:eos_token_id]

    max_length = Nx.axis_size(context.sequences, 1)

    if context.length == max_length - 1 do
      force_token_id(logits, eos_token_id)
    else
      logits
    end
  end

  defn forced_tokens_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:forced_token_ids])
    forced_token_ids(logits, context, opts[:forced_token_ids])
  end

  deftransformp forced_token_ids(logits, context, forced_token_ids) do
    clauses =
      for {idx, token_id} <- forced_token_ids do
        {Nx.equal(context.length, idx), force_token_id(logits, token_id)}
      end

    # Note that we can't use defn ifs inside transform, so we build
    # the expression directly
    Nx.Defn.Expr.cond(clauses, logits)
  end

  defn min_length_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:eos_token_id, :min_length_fun])
    eos_token_id = opts[:eos_token_id]
    min_length_fun = opts[:min_length_fun]

    min_length = min_length_fun.(context.input_length)

    if context.length < min_length do
      ignore_token_id(logits, eos_token_id)
    else
      logits
    end
  end

  defn no_repeat_ngram_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:ngram_length])
    ngram_length = opts[:ngram_length]

    if context.length + 1 < ngram_length do
      logits
    else
      # Given a sequence of last {ngram_length - 1} tokens, we look
      # for prior occurrences of that sequence and we want to make the
      # subsequent token ignored. This way the n-gram is not repeated
      # this time around

      ngram_but_one_length = ngram_length - 1

      last_ngram_but_one =
        Nx.slice_along_axis(
          context.sequences,
          context.length - ngram_but_one_length,
          ngram_but_one_length,
          axis: 1
        )

      {_, _, _, _, logits} =
        while {i = 0, last_ngram_but_one, sequences = context.sequences, length = context.length,
               logits},
              i + ngram_but_one_length < length do
          ngram_but_one = Nx.slice_along_axis(sequences, i, ngram_but_one_length, axis: 1)

          batch_size = Nx.axis_size(logits, 0)

          token_id = sequences[[.., i + ngram_but_one_length]]
          indices = Nx.stack([Nx.iota({batch_size}), token_id], axis: -1)

          match? = Nx.all(ngram_but_one == last_ngram_but_one, axes: [1])
          updates = Nx.select(match?, Nx.Constants.neg_infinity(), 0)

          logits = Nx.indexed_add(logits, indices, updates)

          {i + 1, last_ngram_but_one, sequences, length, logits}
        end

      logits
    end
  end

  deftransformp force_token_id(logits, token_id) do
    batch_size = Nx.axis_size(logits, 0)

    Nx.Constants.neg_infinity()
    |> Nx.broadcast(logits)
    |> Nx.put_slice([0, token_id], Nx.broadcast(0, {batch_size, 1}))
  end

  deftransformp ignore_token_id(logits, token_id) do
    batch_size = Nx.axis_size(logits, 0)

    Nx.put_slice(
      logits,
      [0, token_id],
      Nx.broadcast(Nx.Constants.neg_infinity(), {batch_size, 1})
    )
  end

  # Processors manipulating the probability distribution

  defn top_k_processor(logits, _context, opts \\ []) do
    opts = keyword!(opts, [:top_k])
    top_k = opts[:top_k]

    {top_k_logits, _} = Nx.top_k(logits, k: top_k)
    kth_logit = top_k_logits[[.., -1]]
    Nx.select(logits < kth_logit, Nx.Constants.neg_infinity(), logits)
  end

  defn top_p_processor(logits, _context, opts \\ []) do
    opts = keyword!(opts, [:top_p])
    top_p = opts[:top_p]

    sorted_idx = Nx.argsort(logits, axis: 1)

    cumulative_scores =
      logits
      |> Nx.take_along_axis(sorted_idx, axis: 1)
      |> Axon.Activations.softmax()
      |> Nx.cumulative_sum(axis: 1)

    ordered_ignore_mask = cumulative_scores <= 1 - top_p

    # Arrange the mask back into the original logits order
    ignore_mask =
      Nx.indexed_put(
        Nx.broadcast(0.0, Nx.shape(sorted_idx)),
        Nx.stack([Nx.iota(Nx.shape(sorted_idx), axis: 0), sorted_idx], axis: -1)
        |> Nx.reshape({:auto, 2}),
        Nx.flatten(ordered_ignore_mask)
      )

    Nx.select(ignore_mask, Nx.Constants.neg_infinity(), logits)
  end
end
