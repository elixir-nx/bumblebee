defmodule Bumblebee.Text.DebertaV2 do
  alias Bumblebee.{Layers, Shared}
  alias Bumblebee.Layers.Transformer

  import Nx.Defn

  options = [
    vocab_size: [default: 128_100, doc: "the token vocabulary size"],
    hidden_size: [default: 1536, doc: "the dimensionality of hidden states"],
    embedding_size: [
      default: nil,
      doc: "the embedding dimensionality, defaulting to `:hidden_size`"
    ],
    num_blocks: [default: 24, doc: "the number of encoder blocks"],
    num_attention_heads: [default: 24, doc: "the number of attention heads"],
    attention_head_size: [
      default: nil,
      doc: "the head dimensionality, defaulting to hidden size divided by head count"
    ],
    intermediate_size: [default: 6144, doc: "the feed-forward intermediate dimensionality"],
    activation: [default: :gelu, doc: "the feed-forward activation"],
    dropout_rate: [default: 0.1, doc: "the embedding and residual dropout rate"],
    attention_dropout_rate: [default: 0.1, doc: "the attention probability dropout rate"],
    max_positions: [default: 512, doc: "the maximum number of absolute positions"],
    type_vocab_size: [
      default: 0,
      doc: "the token-type vocabulary size; zero disables token-type embeddings"
    ],
    initializer_scale: [default: 0.02, doc: "the standard deviation of the parameter initializer"],
    layer_norm_epsilon: [default: 1.0e-7, doc: "the layer normalization epsilon"],
    use_relative_attention: [
      default: false,
      doc: "whether to add disentangled relative attention"
    ],
    max_relative_positions: [
      default: -1,
      doc: "the relative position limit; negative values use `:max_positions`"
    ],
    position_buckets: [
      default: -1,
      doc: "the number of logarithmic position buckets; negative values disable bucketing"
    ],
    use_position_embeddings: [default: true, doc: "whether to add absolute position embeddings"],
    position_attention_types: [
      default: [],
      doc:
        "the relative attention terms, a subset of `[:content_to_position, :position_to_content]`"
    ],
    share_attention_key: [
      default: false,
      doc: "whether relative positions reuse content query and key projections"
    ],
    normalize_relative_embeddings: [
      default: false,
      doc: "whether to normalize the relative embedding table"
    ],
    conv_kernel_size: [
      default: 0,
      doc: "the convolution kernel size after the first block; zero disables convolution"
    ],
    conv_groups: [default: 1, doc: "the number of convolution groups"],
    conv_activation: [default: :tanh, doc: "the convolution activation"]
  ]

  @moduledoc """
  DeBERTa-v2 and DeBERTa-v3 base encoder.

  Supports the `:base` architecture. Inputs are `"input_ids"` with shape
  `{batch_size, sequence_length}` and optional `"attention_mask"`,
  `"position_ids"`, and `"token_type_ids"` tensors of the same shape.
  The attention mask defaults to ones; callers must mark padding explicitly.

  Returns `:hidden_state`, and optionally `:hidden_states` and `:attentions`.
  This encoder uses disentangled content/position attention, including
  logarithmic relative position buckets used by DeBERTa-v3.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)
  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  @impl true
  def architectures, do: [:base]

  @impl true
  def config(spec, opts) do
    spec = Shared.put_config_attrs(spec, opts)

    validate_attention_dimensions!(spec)

    if spec.num_blocks < 1, do: raise(ArgumentError, "num_blocks must be positive")

    validate_positions!(spec)

    if spec.conv_kernel_size < 0 or
         (spec.conv_kernel_size > 0 and rem(spec.conv_kernel_size, 2) == 0) do
      raise ArgumentError, "conv_kernel_size must be zero or a positive odd integer"
    end

    spec
  end

  defp validate_attention_dimensions!(spec) do
    if spec.num_attention_heads < 1 or rem(spec.hidden_size, spec.num_attention_heads) != 0 do
      raise ArgumentError, "hidden_size must be divisible by num_attention_heads"
    end

    if spec.attention_head_size != nil and
         spec.attention_head_size * spec.num_attention_heads != spec.hidden_size do
      raise ArgumentError,
            "attention_head_size must equal hidden_size divided by num_attention_heads"
    end
  end

  defp validate_positions!(spec) do
    unless Enum.all?(
             spec.position_attention_types,
             &(&1 in [:content_to_position, :position_to_content])
           ) do
      raise ArgumentError, "unsupported position_attention_types"
    end

    if spec.position_buckets > 0 and
         (spec.position_buckets < 4 or relative_limit(spec) <= div(spec.position_buckets, 2) + 1) do
      raise ArgumentError,
            "position buckets require at least four buckets and a larger relative position limit"
    end
  end

  @impl true
  def input_template(_spec), do: %{"input_ids" => Nx.template({1, 1}, :u32)}

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    ids = Axon.input("input_ids", shape: {nil, nil})

    mask =
      Layers.default(Axon.input("attention_mask", optional: true, shape: {nil, nil}),
        do: Layers.default_attention_mask(ids)
      )

    embedding = embeddings(ids, mask, spec)

    relative = relative_embeddings(spec)

    outputs =
      Transformer.blocks(embedding,
        num_blocks: spec.num_blocks,
        num_attention_heads: spec.num_attention_heads,
        hidden_size: spec.hidden_size,
        attention_mask: mask,
        attention: fn query, _key, _value, opts ->
          {output, weights} = attention(query, relative, mask, spec, opts[:name])
          {output, weights, opts[:attention_cache], Layers.none()}
        end,
        block_output: fn output, index ->
          if index == 0 and spec.conv_kernel_size > 0,
            do: convolution(embedding, output, mask, spec),
            else: output
        end,
        ffn: [intermediate_size: spec.intermediate_size, activation: spec.activation],
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [epsilon: spec.layer_norm_epsilon],
        kernel_initializer: initializer(spec),
        name: "encoder.blocks"
      )

    Layers.output(%{
      hidden_state: outputs.hidden_state,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp attention(hidden, relative, mask, spec, name) do
    relative = Axon.dropout(relative, rate: spec.dropout_rate)
    {query, position_query} = attention_projection(hidden, relative, spec, name <> ".query")
    {key, position_key} = attention_projection(hidden, relative, spec, name <> ".key")

    values =
      hidden
      |> dense(spec.hidden_size, spec, name <> ".value")
      |> Axon.nx(&split_heads(&1, spec.num_attention_heads))

    position_key =
      if spec.use_relative_attention and not spec.share_attention_key and
           :content_to_position in spec.position_attention_types do
        relative
        |> Axon.nx(&Nx.new_axis(&1, 0))
        |> dense(spec.hidden_size, spec, name <> ".position_key")
        |> Axon.nx(&split_heads(&1, spec.num_attention_heads))
      else
        position_key
      end

    position_query =
      if spec.use_relative_attention and not spec.share_attention_key and
           :position_to_content in spec.position_attention_types do
        relative
        |> Axon.nx(&Nx.new_axis(&1, 0))
        |> dense(spec.hidden_size, spec, name <> ".position_query")
        |> Axon.nx(&split_heads(&1, spec.num_attention_heads))
      else
        position_query
      end

    weights =
      Axon.layer(
        &attention_weights/6,
        [query, key, position_query, position_key, mask],
        buckets: spec.position_buckets,
        max_position: relative_limit(spec),
        span: position_span(spec),
        relative: spec.use_relative_attention,
        terms: spec.position_attention_types
      )
      |> Axon.dropout(rate: spec.attention_dropout_rate)

    output =
      Axon.layer(&attention_context/3, [weights, values])
      |> dense(spec.hidden_size, spec, name <> ".output")

    {output, weights}
  end

  defp embeddings(ids, mask, spec) do
    width = spec.embedding_size || spec.hidden_size

    embedding =
      Axon.embedding(ids, spec.vocab_size, width,
        name: "embedder.token_embedding",
        kernel_initializer: initializer(spec)
      )

    embedding =
      if spec.use_position_embeddings do
        positions =
          Layers.default(Axon.input("position_ids", optional: true, shape: {nil, nil}),
            do: Layers.default_position_ids(ids)
          )

        Axon.add(
          embedding,
          Axon.embedding(positions, spec.max_positions, width,
            name: "embedder.position_embedding",
            kernel_initializer: initializer(spec)
          )
        )
      else
        embedding
      end

    embedding =
      if spec.type_vocab_size > 0 do
        types =
          Layers.default(Axon.input("token_type_ids", optional: true, shape: {nil, nil}),
            do: Layers.default_token_type_ids(ids)
          )

        Axon.add(
          embedding,
          Axon.embedding(types, spec.type_vocab_size, width,
            name: "embedder.type_embedding",
            kernel_initializer: initializer(spec)
          )
        )
      else
        embedding
      end

    embedding =
      if width != spec.hidden_size,
        do: dense(embedding, spec.hidden_size, spec, "embedder.projection", use_bias: false),
        else: embedding

    embedding
    |> norm(spec, "embedder.norm")
    |> mask_states(mask)
    |> Axon.dropout(rate: spec.dropout_rate)
  end

  defp relative_embeddings(spec) do
    if spec.use_relative_attention do
      table =
        Axon.layer(
          fn weight, _opts -> weight end,
          [
            Axon.param("kernel", {2 * position_span(spec), spec.hidden_size},
              initializer: initializer(spec)
            )
          ],
          name: "encoder.relative_embedding"
        )

      if spec.normalize_relative_embeddings,
        do: norm(table, spec, "encoder.relative_norm"),
        else: table
    else
      Axon.constant(Nx.tensor(0.0))
    end
  end

  defp attention_projection(hidden, relative, spec, name) do
    Axon.layer(
      &attention_project/5,
      [
        hidden,
        relative,
        Axon.param("kernel", {spec.hidden_size, spec.hidden_size},
          initializer: initializer(spec)
        ),
        Axon.param("bias", {spec.hidden_size}, initializer: :zeros)
      ],
      name: name,
      heads: spec.num_attention_heads,
      relative: spec.use_relative_attention and spec.share_attention_key
    )
    |> Layers.unwrap_tuple(2)
  end

  defp convolution(embedding, residual, mask, spec) do
    embedding
    |> Axon.conv(spec.hidden_size,
      kernel_size: {spec.conv_kernel_size},
      padding: :same,
      feature_group_size: spec.conv_groups,
      kernel_initializer: initializer(spec),
      name: "encoder.conv"
    )
    |> mask_states(mask)
    |> Axon.dropout(rate: spec.dropout_rate)
    |> Layers.activation(spec.conv_activation)
    |> Axon.add(residual)
    |> norm(spec, "encoder.conv_norm")
    |> mask_states(mask)
  end

  defp mask_states(hidden, mask),
    do:
      Axon.layer(fn hidden, mask, _ -> Nx.multiply(hidden, Nx.new_axis(mask, -1)) end, [
        hidden,
        mask
      ])

  defp norm(hidden, spec, name),
    do: Axon.layer_norm(hidden, epsilon: spec.layer_norm_epsilon, name: name)

  defp dense(hidden, units, spec, name, opts \\ []),
    do: Axon.dense(hidden, units, [name: name, kernel_initializer: initializer(spec)] ++ opts)

  defp initializer(spec), do: Axon.Initializers.normal(scale: spec.initializer_scale)

  defp relative_limit(spec),
    do:
      if(spec.max_relative_positions > 0,
        do: spec.max_relative_positions,
        else: spec.max_positions
      )

  defp position_span(spec),
    do: if(spec.position_buckets > 0, do: spec.position_buckets, else: relative_limit(spec))

  defnp attention_project(hidden, relative, kernel, bias, opts) do
    opts = keyword!(opts, [:heads, :relative, :mode])
    content = split_heads(linear(hidden, kernel, bias), opts[:heads])

    position =
      if opts[:relative] do
        split_heads(linear(Nx.new_axis(relative, 0), kernel, bias), opts[:heads])
      else
        Nx.tensor(0.0)
      end

    {content, position}
  end

  defnp attention_weights(query, key, position_query, position_key, mask, opts) do
    opts = keyword!(opts, [:buckets, :max_position, :span, :relative, :terms, :mode])
    scale = Nx.sqrt(Nx.axis_size(query, -1) * (1 + term_count(opts[:terms])))
    scores = product(query, key / scale)

    scores =
      if opts[:relative] do
        positions =
          relative_positions(Nx.axis_size(query, 2), opts[:buckets], opts[:max_position])

        bias = Nx.broadcast(0, Nx.shape(scores))

        bias =
          if has_term?(opts[:terms], :content_to_position) do
            indices = Nx.clip(positions + opts[:span], 0, 2 * opts[:span] - 1)
            bias + gather(product(query, position_key), indices) / scale
          else
            bias
          end

        bias =
          if has_term?(opts[:terms], :position_to_content) do
            indices = Nx.clip(-positions + opts[:span], 0, 2 * opts[:span] - 1)

            bias +
              Nx.transpose(gather(product(key, position_query), indices), axes: [0, 1, 3, 2]) /
                scale
          else
            bias
          end

        scores + bias
      else
        scores
      end

    allowed = Nx.new_axis(Nx.new_axis(mask, 1), 2) * Nx.new_axis(Nx.new_axis(mask, 1), 3)

    scores =
      Nx.select(
        Nx.broadcast(allowed, Nx.shape(scores)),
        scores,
        Nx.Constants.min_finite(Nx.type(scores))
      )

    Axon.Activations.softmax(scores, axis: -1)
  end

  defnp attention_context(weights, value, _opts) do
    {batch, heads, length, width} = Nx.shape(value)

    Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, length, heads * width})
  end

  deftransformp(has_term?(terms, term), do: term in terms)
  deftransformp(term_count(terms), do: length(terms))

  defnp(linear(input, kernel, bias), do: Nx.dot(input, [-1], kernel, [0]) + bias)

  defnp split_heads(input, heads) do
    {batch, length, hidden} = Nx.shape(input)

    input
    |> Nx.reshape({batch, length, heads, div(hidden, heads)})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defnp product(left, right) do
    right =
      Nx.broadcast(
        right,
        {Nx.axis_size(left, 0), Nx.axis_size(right, 1), Nx.axis_size(right, 2),
         Nx.axis_size(right, 3)}
      )

    Nx.dot(left, [3], [0, 1], right, [3], [0, 1])
  end

  defnp gather(scores, positions) do
    shape =
      {Nx.axis_size(scores, 0), Nx.axis_size(scores, 1), Nx.axis_size(positions, 0),
       Nx.axis_size(positions, 1)}

    Nx.take_along_axis(scores, Nx.broadcast(positions, shape), axis: 3)
  end

  defnp relative_positions(length, buckets, max_position) do
    ids = Nx.iota({length}, type: :s64)
    relative = Nx.new_axis(ids, 1) - Nx.new_axis(ids, 0)

    if buckets > 0 do
      mid = div(buckets, 2)
      absolute = Nx.select(Nx.abs(relative) < mid, mid - 1, Nx.abs(relative))

      logarithmic =
        Nx.ceil(Nx.log(absolute / mid) / Nx.log((max_position - 1) / mid) * (mid - 1)) + mid

      Nx.select(absolute <= mid, relative, logarithmic * Nx.sign(relative)) |> Nx.as_type(:s64)
    else
      relative
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          hidden_size: {"hidden_size", number()},
          embedding_size: {"embedding_size", optional(number())},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          attention_head_size: {"attention_head_size", optional(number())},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          max_positions: {"max_position_embeddings", number()},
          type_vocab_size: {"type_vocab_size", number()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          use_relative_attention: {"relative_attention", boolean()},
          max_relative_positions: {"max_relative_positions", number()},
          position_buckets: {"position_buckets", number()},
          use_position_embeddings: {"position_biased_input", boolean()},
          share_attention_key: {"share_att_key", boolean()},
          conv_kernel_size: {"conv_kernel_size", number()},
          conv_groups: {"conv_groups", number()},
          conv_activation: {"conv_act", activation()}
        )

      terms =
        case data["pos_att_type"] do
          nil -> []
          value when is_binary(value) -> String.split(value, "|", trim: true)
          value -> value
        end

      terms =
        Enum.flat_map(terms, fn term ->
          case String.trim(term) do
            "c2p" -> [:content_to_position]
            "p2c" -> [:position_to_content]
            "none" -> []
            other -> raise ArgumentError, "unsupported position attention type: #{inspect(other)}"
          end
        end)

      normalize = "layer_norm" in String.split(data["norm_rel_ebd"] || "none", "|")

      @for.config(
        spec,
        opts ++ [position_attention_types: terms, normalize_relative_embeddings: normalize]
      )
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "deberta.embeddings.word_embeddings",
        "embedder.position_embedding" => "deberta.embeddings.position_embeddings",
        "embedder.type_embedding" => "deberta.embeddings.token_type_embeddings",
        "embedder.projection" => "deberta.embeddings.embed_proj",
        "embedder.norm" => "deberta.embeddings.LayerNorm",
        "encoder.relative_embedding" => %{
          "kernel" => {[{"deberta.encoder.rel_embeddings", "weight"}], fn [tensor] -> tensor end}
        },
        "encoder.relative_norm" => "deberta.encoder.LayerNorm",
        "encoder.blocks.{n}.self_attention.query" => %{
          "kernel" =>
            {[{"deberta.encoder.layer.{n}.attention.self.query_proj", "weight"}],
             fn [tensor] -> Nx.transpose(tensor) end},
          "bias" =>
            {[{"deberta.encoder.layer.{n}.attention.self.query_proj", "bias"}],
             fn [tensor] -> tensor end}
        },
        "encoder.blocks.{n}.self_attention.key" => %{
          "kernel" =>
            {[{"deberta.encoder.layer.{n}.attention.self.key_proj", "weight"}],
             fn [tensor] -> Nx.transpose(tensor) end},
          "bias" =>
            {[{"deberta.encoder.layer.{n}.attention.self.key_proj", "bias"}],
             fn [tensor] -> tensor end}
        },
        "encoder.blocks.{n}.self_attention.value" =>
          "deberta.encoder.layer.{n}.attention.self.value_proj",
        "encoder.blocks.{n}.self_attention.position_key" =>
          "deberta.encoder.layer.{n}.attention.self.pos_key_proj",
        "encoder.blocks.{n}.self_attention.position_query" =>
          "deberta.encoder.layer.{n}.attention.self.pos_query_proj",
        "encoder.blocks.{n}.self_attention.output" =>
          "deberta.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.self_attention_norm" =>
          "deberta.encoder.layer.{n}.attention.output.LayerNorm",
        "encoder.blocks.{n}.ffn.intermediate" => "deberta.encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => "deberta.encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => "deberta.encoder.layer.{n}.output.LayerNorm",
        "encoder.conv" => "deberta.encoder.conv.conv",
        "encoder.conv_norm" => "deberta.encoder.conv.LayerNorm"
      }
    end
  end
end
