defmodule Bumblebee.Text.Gliner.CandidatePool do
  @moduledoc false

  def select(output, config) do
    starts = rows(output["start_logits"])
    endings = rows(output["end_logits"])
    union_start = starts |> Enum.zip_with(&Enum.max/1) |> List.to_tuple()
    union_end = endings |> Enum.zip_with(&Enum.max/1) |> List.to_tuple()
    top_start = top_indices(union_start, config["pool_boundary_top_k"])
    top_end = top_indices(union_end, config["pool_boundary_top_k"])
    compatibilities = compatibility(output, top_start, top_end)

    pairs =
      for {start, i} <- Enum.with_index(top_start),
          {ending, j} <- Enum.with_index(top_end),
          ending > start do
        %{start: start, end: ending, compat: elem(compatibilities, i * length(top_end) + j)}
      end

    global =
      Enum.map(
        pairs,
        &Map.put(&1, :priority, &1.compat + elem(union_start, &1.start) + elem(union_end, &1.end))
      )

    quota = config["min_pool_per_query"]

    reserved =
      Enum.zip_with(starts, endings, fn start_logits, end_logits ->
        start_logits = List.to_tuple(start_logits)
        end_logits = List.to_tuple(end_logits)

        pairs
        |> Enum.sort_by(&(-(&1.compat + elem(start_logits, &1.start) + elem(end_logits, &1.end))))
        |> Enum.take(quota)
        |> Enum.with_index(fn pair, rank -> Map.put(pair, :priority, 5000.0 + quota - rank) end)
      end)
      |> List.flatten()

    selected =
      (reserved ++ global)
      |> Enum.sort_by(&(-&1.priority))
      |> Enum.uniq_by(&{&1.start, &1.end})
      |> Enum.sort_by(&{-&1.priority, &1.start, &1.end})
      |> Enum.take(config["pool_size"])

    padding = config["pool_size"] - length(selected)

    %{
      "starts" =>
        Nx.tensor([Enum.map(selected, & &1.start) ++ List.duplicate(0, padding)], type: :s64),
      "ends" =>
        Nx.tensor([Enum.map(selected, & &1.end) ++ List.duplicate(0, padding)], type: :s64),
      "compat" =>
        Nx.tensor([Enum.map(selected, & &1.compat) ++ List.duplicate(0.0, padding)], type: :f32),
      "valid" =>
        Nx.tensor([List.duplicate(1, length(selected)) ++ List.duplicate(0, padding)], type: :u8)
    }
  end

  defp rows(tensor) do
    tensor |> Nx.to_flat_list() |> Enum.chunk_every(Nx.axis_size(tensor, -1))
  end

  defp top_indices(logits, count) do
    logits
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.sort_by(fn {score, index} -> {-score, index} end)
    |> Enum.take(count)
    |> Enum.map(&elem(&1, 1))
  end

  defp compatibility(output, starts, endings) do
    left = Nx.take(output["pool_start"], Nx.tensor(starts), axis: 1) |> Nx.squeeze(axes: [0])
    right = Nx.take(output["pool_end"], Nx.tensor(endings), axis: 1) |> Nx.squeeze(axes: [0])

    Nx.multiply(Nx.new_axis(left, 1), Nx.new_axis(right, 0))
    |> Nx.sum(axes: [2])
    |> Nx.divide(:math.sqrt(Nx.axis_size(left, 1)))
    |> Nx.to_flat_list()
    |> List.to_tuple()
  end
end
