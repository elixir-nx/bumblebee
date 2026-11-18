defmodule Bumblebee.Text.Gliner.PairScorer do
  @moduledoc false
  import Nx.Defn
  @prefix "boundary_head.shared_pool_scorer."

  def model(config, hidden_size) do
    dim = config["pair_dim"]
    states = Axon.input("boundary_states", shape: {1, nil, config["boundary_dim"]})
    query = Axon.input("query_states", shape: {1, nil, hidden_size})
    text = Axon.input("text_states", shape: {1, nil, hidden_size})
    starts = Axon.input("starts", shape: {1, nil})
    endings = Axon.input("ends", shape: {1, nil})
    compat = Axon.input("compat", shape: {1, nil})
    valid = Axon.input("valid", shape: {1, nil})
    start_logits = Axon.input("start_logits", shape: {1, nil, nil})
    end_logits = Axon.input("end_logits", shape: {1, nil, nil})
    inside_logits = Axon.input("inside_logits", shape: {1, nil, nil})
    start_rep = gather(dense(states, dim, "start_projection"), starts)
    end_rep = gather(dense(states, dim, "end_projection"), endings)

    lengths =
      Axon.layer(&length_features/4, [starts, endings, text]) |> dense(dim, "length_projection")

    prior = compat |> Axon.nx(&Nx.new_axis(&1, -1)) |> dense(dim, "prior_projection")
    values = dense(text, config["content_dim"], "content_pooler.value_projection")

    content =
      Axon.layer(&content_mean/4, [values, starts, endings])
      |> Axon.layer_norm(name: @prefix <> "content_pooler.layer_norm")
      |> dense(dim, "content_projection")

    candidate =
      Axon.add([start_rep, end_rep, lengths, prior, content])
      |> Axon.layer_norm(name: @prefix <> "candidate_norm")

    candidate = Axon.layer(&mask_states/3, [candidate, valid])
    projected_query = dense(query, dim, "query_projection")
    film = dense(projected_query, 2 * dim, "film")

    conditioned =
      Axon.layer(&condition/3, [candidate, film])
      |> dense(64, "film_output.0")
      |> Axon.gelu()
      |> dense(1, "film_output.3")

    score =
      Axon.layer(&score/7, [
        candidate,
        projected_query,
        conditioned,
        start_logits,
        end_logits,
        Axon.container({starts, endings})
      ])

    score = Axon.layer(&inside_evidence/5, [score, inside_logits, starts, endings])
    Axon.layer(&mask_scores/3, [score, valid])
  end

  defp dense(input, units, name), do: Axon.dense(input, units, name: @prefix <> name)
  defp gather(input, indices), do: Axon.layer(&gather_rows/3, [input, indices])

  defnp(gather_rows(input, indices, _opts),
    do: Nx.take(input, Nx.squeeze(indices, axes: [0]), axis: 1)
  )

  defnp(mask_states(states, valid, _opts), do: states * Nx.new_axis(valid, -1))

  defnp length_features(starts, endings, text, _opts) do
    length = Nx.as_type(Nx.max(endings - starts, 1), :f32)
    Nx.stack([Nx.log1p(length), length / Nx.axis_size(text, 1), Nx.rsqrt(length)], axis: -1)
  end

  defnp content_mean(values, starts, endings, _opts) do
    prefix =
      Nx.concatenate(
        [
          Nx.broadcast(0.0, {1, 1, Nx.axis_size(values, 2)}),
          Nx.cumulative_sum(Nx.as_type(values, :f64), axis: 1) |> Nx.as_type(:f32)
        ],
        axis: 1
      )

    (gather_rows(prefix, endings, []) - gather_rows(prefix, starts, [])) /
      Nx.new_axis(Nx.max(endings - starts, 1), -1)
  end

  defnp condition(candidate, film, _opts) do
    {gamma, beta} = Nx.split(film, 0.5, axis: -1)
    Nx.new_axis(candidate, 2) * (1.0 + Nx.new_axis(gamma, 1)) + Nx.new_axis(beta, 1)
  end

  defnp score(candidate, query, conditioned, start_logits, end_logits, indices, _opts) do
    {starts, endings} = indices
    base = Nx.dot(candidate, [2], [0], query, [2], [0]) / Nx.sqrt(Nx.axis_size(candidate, 2))

    start_scores =
      Nx.take(start_logits, Nx.squeeze(starts, axes: [0]), axis: 2)
      |> Nx.transpose(axes: [0, 2, 1])

    end_scores =
      Nx.take(end_logits, Nx.squeeze(endings, axes: [0]), axis: 2)
      |> Nx.transpose(axes: [0, 2, 1])

    base + Nx.squeeze(conditioned, axes: [3]) + start_scores + end_scores
  end

  defnp inside_evidence(scores, inside, starts, endings, _opts) do
    mean = Nx.mean(inside, axes: [2], keep_axes: true)

    prefix =
      Nx.concatenate(
        [
          Nx.broadcast(0.0, {1, Nx.axis_size(inside, 1), 1}),
          Nx.cumulative_sum(Nx.as_type(inside - mean, :f64), axis: 2) |> Nx.as_type(:f32)
        ],
        axis: 2
      )

    start_prefix = Nx.take(prefix, Nx.squeeze(starts, axes: [0]), axis: 2)
    end_prefix = Nx.take(prefix, Nx.squeeze(endings, axes: [0]), axis: 2)
    length = Nx.new_axis(endings - starts, 1)
    evidence = (end_prefix - start_prefix + mean * length) / Nx.sqrt(Nx.max(length, 1))
    scores + Nx.transpose(evidence, axes: [0, 2, 1])
  end

  defnp mask_scores(scores, valid, _opts) do
    Nx.select(Nx.broadcast(Nx.new_axis(valid, -1), Nx.shape(scores)), scores, -1.0e4)
    |> Nx.transpose(axes: [0, 2, 1])
  end
end
