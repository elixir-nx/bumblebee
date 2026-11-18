defmodule Bumblebee.Text.Gliner.BoundaryHead do
  @moduledoc false
  import Nx.Defn

  def model(config, hidden_size) do
    text = Axon.input("text_states", shape: {1, nil, hidden_size})
    query = Axon.input("query_states", shape: {1, nil, hidden_size})
    dim = config["boundary_dim"]
    prefix = "boundary_head.boundary_encoder"

    left =
      Axon.layer(&left_states/3, [text, Axon.param("bos_state", {hidden_size})],
        name: prefix <> ".bos"
      )

    right =
      Axon.layer(&right_states/3, [text, Axon.param("eos_state", {hidden_size})],
        name: prefix <> ".eos"
      )

    left = Axon.dense(left, dim, name: prefix <> ".left_projection")
    right = Axon.dense(right, dim, name: prefix <> ".right_projection")

    boundary =
      Axon.concatenate([left, right])
      |> Axon.dense(dim, name: prefix <> ".output_projection")
      |> Axon.layer_norm(name: prefix <> ".layer_norm")

    boundary =
      Enum.reduce(0..(config["boundary_attention_layers"] - 1)//1, boundary, fn index, states ->
        name = prefix <> ".attention_blocks.#{index}"

        projected =
          states
          |> Axon.layer_norm(name: name <> ".norm")
          |> Axon.dense(3 * dim, name: name <> ".qkv_projection")

        update =
          Axon.layer(&self_attention/2, [projected],
            heads: config["boundary_attention_heads"],
            window: config["boundary_attention_window"]
          )
          |> Axon.dense(dim, name: name <> ".output_projection")

        Axon.add(states, update)
      end)

    boundary =
      Enum.reduce(0..(config["boundary_refinement_layers"] - 1)//1, boundary, fn index, states ->
        name = prefix <> ".refinement_blocks.#{index}"
        inner = max(1, trunc(dim * config["boundary_ffn_multiplier"]))

        update =
          states
          |> Axon.layer_norm(name: name <> ".norm")
          |> Axon.dense(2 * inner, name: name <> ".input_projection")
          |> Axon.nx(&swiglu/1)
          |> Axon.dense(dim, name: name <> ".output_projection")

        Axon.add(states, update)
      end)

    start = marginal(boundary, query, dim, "start_boundary_projection", "start_query_projection")
    ending = marginal(boundary, query, dim, "end_boundary_projection", "end_query_projection")
    inside = marginal(text, query, dim, "inside_text_projection", "inside_query_projection")
    null = Axon.dense(query, 1, name: "boundary_head.null_projection")

    Axon.container(%{
      "boundary_states" => boundary,
      "start_logits" => start,
      "end_logits" => ending,
      "inside_logits" => inside,
      "null_logits" => null,
      "pool_start" =>
        Axon.dense(boundary, dim, name: "boundary_head.shared_pool_builder.start_projection"),
      "pool_end" =>
        Axon.dense(boundary, dim, name: "boundary_head.shared_pool_builder.end_projection")
    })
  end

  defp marginal(states, query, dim, state_name, query_name) do
    prefix = "boundary_head.boundary_query_head."
    projected_states = Axon.dense(states, dim, name: prefix <> state_name)
    projected_query = Axon.dense(query, dim, name: prefix <> query_name)
    Axon.layer(&query_scores/3, [projected_states, projected_query])
  end

  defnp(left_states(text, bos, _opts),
    do: Nx.concatenate([Nx.reshape(bos, {1, 1, Nx.axis_size(bos, 0)}), text], axis: 1)
  )

  defnp(right_states(text, eos, _opts),
    do: Nx.concatenate([text, Nx.reshape(eos, {1, 1, Nx.axis_size(eos, 0)})], axis: 1)
  )

  defn query_scores(states, query, _opts) do
    Nx.dot(query, [2], [0], states, [2], [0]) / Nx.sqrt(Nx.axis_size(states, 2))
  end

  defnp swiglu(states) do
    {value, gate} = Nx.split(states, 0.5, axis: -1)
    value * Axon.Activations.silu(gate)
  end

  defn self_attention(projected, opts) do
    opts = keyword!(opts, heads: 4, window: 128, mode: :inference)
    {batch, length, triple_dim} = Nx.shape(projected)
    dim = div(triple_dim, 3)
    heads = opts[:heads]
    width = div(dim, heads)

    qkv =
      projected
      |> Nx.reshape({batch, length, 3, heads, width})
      |> Nx.transpose(axes: [2, 0, 3, 1, 4])

    query = qkv[0]
    key = qkv[1]
    value = qkv[2]
    scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1]) / Nx.sqrt(width)
    ids = Nx.iota({length})

    allowed =
      if opts[:window] > 0,
        do: Nx.abs(Nx.new_axis(ids, 1) - Nx.new_axis(ids, 0)) <= opts[:window],
        else: Nx.broadcast(1, {length, length})

    scores = Nx.select(Nx.broadcast(allowed, Nx.shape(scores)), scores, -1.0e4)
    weights = Axon.Activations.softmax(scores, axis: -1)

    Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, length, dim})
  end
end
