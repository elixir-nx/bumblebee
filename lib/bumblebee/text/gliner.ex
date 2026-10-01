defmodule Bumblebee.Text.Gliner do
  alias Bumblebee.Shared
  alias Bumblebee.Text.{DebertaV2, Gliner}

  @moduledoc """
  GLiNER 2.5 boundary-based named entity extraction model stages.

  The `:encoder` stage wraps `Bumblebee.Text.DebertaV2`; `:boundary` computes
  boundary states and marginals from routed text and query states; and
  `:span_scoring` scores a host-selected shared candidate pool. These stages
  support inference with one unpadded document and an arbitrary number of labels.

  Load the nested encoder configuration from `encoder_config/config.json`
  and set `:encoder_spec` when loading `:encoder`. Set `:hidden_size` on the
  other stages to match that encoder. The default hidden size is 768.

  This implementation supports first-token word pooling, a shared candidate
  pool, span content, inside evidence, abstention and flat overlap resolution.
  Unsupported architectural variants raise during configuration. Classification,
  records and relations are outside this model's extraction API.

  Boundary configuration options use the names in the checkpoint's
  `boundary_head` object. Model outputs use the same string keys as their
  downstream stage inputs. `:span_scoring` returns a tensor shaped
  `{1, number_of_labels, number_of_candidates}`.
  """

  defstruct architecture: :boundary,
            encoder_spec: nil,
            hidden_size: 768,
            abstention_threshold: 0.5,
            adaptive_threshold: false,
            boundary_attention_heads: 4,
            boundary_attention_layers: 2,
            boundary_attention_window: 128,
            boundary_dim: 128,
            boundary_ffn_multiplier: 2.0,
            boundary_refinement_layers: 1,
            candidate_attention_layers: 0,
            candidate_pool: "shared",
            content_dim: 64,
            content_soft_max_pool: false,
            enable_abstention: true,
            enable_span_content: true,
            min_pool_per_query: 8,
            overlap_policy: "flat",
            pair_dim: 128,
            pair_temperature: 1.0,
            pool_boundary_top_k: 32,
            pool_size: 192,
            query_attention_layers: 0,
            use_inside_evidence: true

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures, do: [:encoder, :boundary, :span_scoring]

  @impl true
  def config(spec, opts) do
    spec = Shared.put_config_attrs(spec, opts)

    validate_architecture!(spec)
    validate_dimensions!(spec)
    validate_scoring!(spec)

    spec
  end

  defp validate_architecture!(spec) do
    required = [
      candidate_pool: "shared",
      candidate_attention_layers: 0,
      query_attention_layers: 0,
      content_soft_max_pool: false,
      enable_span_content: true,
      enable_abstention: true,
      use_inside_evidence: true,
      adaptive_threshold: false,
      overlap_policy: "flat"
    ]

    for {key, value} <- required, Map.fetch!(spec, key) != value do
      raise ArgumentError, "unsupported GLiNER option #{key}: #{inspect(Map.fetch!(spec, key))}"
    end
  end

  defp validate_dimensions!(spec) do
    for key <- [
          :hidden_size,
          :boundary_dim,
          :boundary_attention_heads,
          :pair_dim,
          :content_dim,
          :pool_size,
          :pool_boundary_top_k
        ] do
      value = Map.fetch!(spec, key)

      unless is_integer(value) and value > 0,
        do: raise(ArgumentError, "#{key} must be a positive integer")
    end

    unless rem(spec.boundary_dim, spec.boundary_attention_heads) == 0,
      do: raise(ArgumentError, "boundary_dim must be divisible by boundary_attention_heads")

    for key <- [
          :boundary_attention_layers,
          :boundary_refinement_layers,
          :boundary_attention_window,
          :min_pool_per_query
        ] do
      value = Map.fetch!(spec, key)

      unless is_integer(value) and value >= 0,
        do: raise(ArgumentError, "#{key} must be a nonnegative integer")
    end
  end

  defp validate_scoring!(spec) do
    for key <- [:boundary_ffn_multiplier, :pair_temperature] do
      value = Map.fetch!(spec, key)
      unless is_number(value) and value > 0, do: raise(ArgumentError, "#{key} must be positive")
    end

    unless is_number(spec.abstention_threshold) and spec.abstention_threshold >= 0 and
             spec.abstention_threshold <= 1 do
      raise ArgumentError, "abstention_threshold must be between zero and one"
    end
  end

  @doc false
  def boundary_config(spec) do
    spec
    |> Map.from_struct()
    |> Map.drop([:architecture, :encoder_spec, :hidden_size])
    |> Map.new(fn {key, value} -> {Atom.to_string(key), value} end)
  end

  @impl true
  def input_template(%{architecture: :encoder, encoder_spec: %DebertaV2{} = encoder}),
    do: DebertaV2.input_template(encoder)

  def input_template(%{architecture: :boundary} = spec) do
    %{
      "text_states" => Nx.template({1, 2, spec.hidden_size}, :f32),
      "query_states" => Nx.template({1, 1, spec.hidden_size}, :f32)
    }
  end

  def input_template(%{architecture: :span_scoring} = spec) do
    Map.merge(input_template(%{spec | architecture: :boundary}), %{
      "boundary_states" => Nx.template({1, 3, spec.boundary_dim}, :f32),
      "starts" => Nx.template({1, spec.pool_size}, :s64),
      "ends" => Nx.template({1, spec.pool_size}, :s64),
      "compat" => Nx.template({1, spec.pool_size}, :f32),
      "valid" => Nx.template({1, spec.pool_size}, :u8),
      "start_logits" => Nx.template({1, 1, 3}, :f32),
      "end_logits" => Nx.template({1, 1, 3}, :f32),
      "inside_logits" => Nx.template({1, 1, 2}, :f32)
    })
  end

  @impl true
  def model(%{architecture: :encoder, encoder_spec: %DebertaV2{} = encoder}),
    do: DebertaV2.model(encoder)

  def model(%{architecture: :encoder}),
    do: raise(ArgumentError, "encoder_spec must be a configured DebertaV2 encoder")

  def model(%{architecture: :boundary} = spec),
    do: Gliner.BoundaryHead.model(boundary_config(spec), spec.hidden_size)

  def model(%{architecture: :span_scoring} = spec),
    do: Gliner.PairScorer.model(boundary_config(spec), spec.hidden_size)

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      unless data["architecture"] == "boundary" and data["token_pooling"] == "first" do
        raise ArgumentError, "expected a GLiNER boundary checkpoint with first-token pooling"
      end

      boundary = Map.fetch!(data, "boundary_head")

      opts =
        for {key, _} <- @for.boundary_config(spec),
            Map.has_key?(boundary, key),
            do: {String.to_existing_atom(key), boundary[key]}

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.HuggingFace.Transformers
    alias Bumblebee.Utils

    def params_mapping(%{architecture: :encoder, encoder_spec: encoder}) do
      Map.new(Transformers.Model.params_mapping(encoder), fn {target, source} ->
        {target,
         Transformers.Utils.map_params_source_layer_names(
           source,
           &String.replace_prefix(&1, "deberta.", "encoder.")
         )}
      end)
    end

    def params_mapping(spec) do
      for {node, name} <- @for.model(spec) |> Utils.Axon.nodes_with_names(),
          node.parameters != [],
          into: %{} do
        cond do
          String.ends_with?(name, ".bos") ->
            {name,
             %{
               "bos_state" =>
                 {[{String.trim_trailing(name, ".bos"), "bos_state"}], fn [value] -> value end}
             }}

          String.ends_with?(name, ".eos") ->
            {name,
             %{
               "eos_state" =>
                 {[{String.trim_trailing(name, ".eos"), "eos_state"}], fn [value] -> value end}
             }}

          true ->
            {name, name}
        end
      end
    end
  end
end
