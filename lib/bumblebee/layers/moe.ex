defmodule Bumblebee.Layers.Moe do
  @moduledoc false

  # Layers implementing sparse Mixture-of-Experts (MoE) feed-forward blocks,
  # as used by DeepSeek, GLM, Kimi and Inkling model families.
  #
  # A MoE block replaces the single feed-forward network of a Transformer
  # block with `num_experts` independent networks (the experts) plus a
  # router. The router scores every expert for every token and only a
  # handful of the highest scoring experts (`num_experts_per_token`)
  # contribute to the output, which keeps the number of active parameters
  # per token low, while the total number of parameters is large.
  #
  # ## Implementation notes
  #
  # The expert weights are stored stacked, that is, a single parameter of
  # shape `{num_experts, ...}` rather than a separate parameter per expert.
  # The mixture is then computed densely: every expert is applied to every
  # token and the results are combined using the router weights, which are
  # zero for the experts that were not selected. This trades extra compute
  # (proportional to `num_experts / num_experts_per_token`) for a static
  # computation graph with no gather/scatter, which is what XLA is good at.

  import Nx.Defn

  alias Bumblebee.Layers

  import Bumblebee.Utils.Model, only: [join: 2]

  @doc """
  Adds a mixture-of-experts feed-forward block.

  The block consists of a router, a set of routed experts and an optional
  shared expert, which is applied to all tokens.

  ## Options

    * `:num_experts` (required) - the number of routed experts

    * `:num_experts_per_token` (required) - the number of experts that
      each token is routed to

    * `:hidden_size` (required) - the dimensionality of the input and
      output

    * `:intermediate_size` (required) - the dimensionality of each expert

    * `:activation` - the activation used within each expert. Defaults
      to `:silu`

    * `:shared_expert_intermediate_size` - when set, adds a shared expert
      (a regular gated feed-forward network) with the given intermediate
      size, applied to all tokens and added to the routed output

    * `:name` - the prefix for layer names

  For the remaining options see `router/2`.
  """
  def block(hidden_state, opts) do
    validate_required_keys!(opts, [
      :num_experts,
      :num_experts_per_token,
      :hidden_size,
      :intermediate_size
    ])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_experts,
        :num_experts_per_token,
        :hidden_size,
        :intermediate_size,
        :shared_expert_intermediate_size,
        :expert_group,
        activation: :silu,
        scoring: :sigmoid,
        normalize_top_k: true,
        routed_scaling_factor: 1.0,
        use_score_correction_bias: true,
        use_bias: false,
        logit_softcapping: nil,
        expert_use_bias: false,
        expert_variant: :gated,
        expert_clamp_limit: 7.0,
        expert_activation_alpha: 1.702,
        kernel_initializer: :glorot_uniform
      ])

    name = opts[:name]

    weights =
      router(hidden_state,
        num_experts: opts[:num_experts],
        num_experts_per_token: opts[:num_experts_per_token],
        expert_group: opts[:expert_group],
        scoring: opts[:scoring],
        normalize_top_k: opts[:normalize_top_k],
        routed_scaling_factor: opts[:routed_scaling_factor],
        use_score_correction_bias: opts[:use_score_correction_bias],
        use_bias: opts[:use_bias],
        logit_softcapping: opts[:logit_softcapping],
        kernel_initializer: opts[:kernel_initializer],
        name: join(name, "router")
      )

    routed =
      experts(hidden_state, weights,
        num_experts: opts[:num_experts],
        hidden_size: opts[:hidden_size],
        intermediate_size: opts[:intermediate_size],
        activation: opts[:activation],
        use_bias: opts[:expert_use_bias],
        variant: opts[:expert_variant],
        clamp_limit: opts[:expert_clamp_limit],
        activation_alpha: opts[:expert_activation_alpha],
        kernel_initializer: opts[:kernel_initializer],
        name: join(name, "experts")
      )

    case opts[:shared_expert_intermediate_size] do
      nil ->
        routed

      intermediate_size ->
        shared =
          if opts[:expert_variant] == :ungated do
            ffn(hidden_state, intermediate_size, opts[:hidden_size],
              activation: opts[:activation],
              kernel_initializer: opts[:kernel_initializer],
              name: join(name, "shared_expert")
            )
          else
            gated_ffn(hidden_state, intermediate_size, opts[:hidden_size],
              activation: opts[:activation],
              kernel_initializer: opts[:kernel_initializer],
              name: join(name, "shared_expert")
            )
          end

        Axon.add(routed, shared)
    end
  end

  @doc """
  Adds an expert router.

  Returns a node with the routing weights, a tensor of shape
  `{batch_size, sequence_length, num_experts}`, where all but
  `:num_experts_per_token` entries are zero for every token.

  ## Options

    * `:num_experts` (required) - the number of routed experts

    * `:num_experts_per_token` (required) - the number of experts that
      each token is routed to

    * `:scoring` - the function converting router logits into scores.
      One of `:sigmoid`, `:softmax`, `:sqrt_softplus`, or `:softmax_top_k`,
      where the latter applies the softmax over the selected experts only.
      Defaults to `:sigmoid`

    * `:use_score_correction_bias` - whether the router has a per-expert
      bias, added to the scores for the purpose of expert selection only.
      Defaults to `true`

    * `:expert_group` - when set, enables group-limited routing, where
      the experts are split into groups, only a few groups are considered
      for each token and the experts are then selected within those groups.
      Expects a keyword list with `:num_groups` and `:num_groups_per_token`

    * `:normalize_top_k` - whether to normalize the weights of the
      selected experts to sum up to one. Defaults to `true`

    * `:routed_scaling_factor` - a constant that the weights are
      multiplied by. Defaults to `1.0`

    * `:name` - the prefix for layer names

  """
  def router(hidden_state, opts) do
    validate_required_keys!(opts, [:num_experts, :num_experts_per_token])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_experts,
        :num_experts_per_token,
        :expert_group,
        scoring: :sigmoid,
        normalize_top_k: true,
        routed_scaling_factor: 1.0,
        use_score_correction_bias: true,
        use_bias: false,
        logit_softcapping: nil,
        kernel_initializer: :glorot_uniform
      ])

    name = opts[:name]
    num_experts = opts[:num_experts]

    {num_groups, num_groups_per_token} =
      case opts[:expert_group] do
        nil ->
          {1, 1}

        group_opts ->
          group_opts =
            Keyword.validate!(group_opts, [:num_groups, :num_groups_per_token])

          {group_opts[:num_groups], group_opts[:num_groups_per_token]}
      end

    kernel =
      Axon.param("kernel", fn shape -> {elem(shape, tuple_size(shape) - 1), num_experts} end,
        initializer: opts[:kernel_initializer]
      )

    {inputs, impl} =
      cond do
        opts[:use_bias] ->
          bias = Axon.param("bias", fn _ -> {num_experts} end, initializer: :zeros)
          {[hidden_state, kernel, bias], &biased_router_impl/4}

        opts[:use_score_correction_bias] ->
          bias =
            Axon.param("score_correction_bias", fn _ -> {num_experts} end, initializer: :zeros)

          {[hidden_state, kernel, bias], &biased_router_impl/4}

        true ->
          {[hidden_state, kernel], &router_impl/3}
      end

    Axon.layer(impl, inputs,
      name: name,
      op_name: :moe_router,
      num_experts: num_experts,
      num_experts_per_token: opts[:num_experts_per_token],
      num_groups: num_groups,
      num_groups_per_token: num_groups_per_token,
      scoring: opts[:scoring],
      normalize_top_k: opts[:normalize_top_k],
      routed_scaling_factor: opts[:routed_scaling_factor],
      logit_bias: opts[:use_bias],
      logit_softcapping: opts[:logit_softcapping]
    )
  end

  defnp router_impl(hidden_state, kernel, opts \\ []) do
    biased_router_impl(hidden_state, kernel, 0.0, opts)
  end

  defnp biased_router_impl(hidden_state, kernel, bias, opts \\ []) do
    opts =
      keyword!(opts, [
        :num_experts,
        :num_experts_per_token,
        :num_groups,
        :num_groups_per_token,
        :scoring,
        :normalize_top_k,
        :routed_scaling_factor,
        :logit_softcapping,
        logit_bias: false,
        mode: :inference
      ])

    num_experts_per_token = opts[:num_experts_per_token]

    # The router is computed in high precision, since the selection is
    # discrete and sensitive to rounding
    hidden_state = Nx.as_type(hidden_state, :f32)
    kernel = Nx.as_type(kernel, :f32)
    bias = Nx.as_type(bias, :f32)

    logits = Nx.dot(hidden_state, [-1], kernel, [0])

    # Some routers cap the logits before scoring
    logits =
      case opts[:logit_softcapping] do
        nil -> logits
        cap -> Nx.tanh(logits / cap) * cap
      end

    logits = if opts[:logit_bias], do: logits + bias, else: logits

    case opts[:scoring] do
      :softmax_top_k ->
        # The softmax is computed over the selected experts only
        mask = top_k_mask(logits, num_experts_per_token)

        selected_logits =
          Nx.select(mask, logits, Nx.Constants.neg_infinity(:f32))

        denominator = Nx.logsumexp(selected_logits, axes: [-1], keep_axes: true)

        Nx.exp(logits - denominator) * mask

      scoring ->
        scored_router(logits, scoring, if(opts[:logit_bias], do: 0.0, else: bias), opts)
    end
  end

  defnp scored_router(logits, scoring, score_correction_bias, opts) do
    num_experts_per_token = opts[:num_experts_per_token]
    num_groups = opts[:num_groups]
    num_groups_per_token = opts[:num_groups_per_token]

    scores =
      case scoring do
        :sigmoid -> Nx.sigmoid(logits)
        :softmax -> Axon.Activations.softmax(logits, axis: -1)
        :sqrt_softplus -> Nx.sqrt(Axon.Activations.softplus(logits))
      end

    selection_scores = scores + score_correction_bias

    selection_scores =
      if num_groups > 1 do
        mask_expert_groups(selection_scores, num_groups, num_groups_per_token)
      else
        selection_scores
      end

    mask = top_k_mask(selection_scores, num_experts_per_token)

    weights = scores * mask

    weights =
      if opts[:normalize_top_k] do
        weights / (Nx.sum(weights, axes: [-1], keep_axes: true) + 1.0e-20)
      else
        weights
      end

    weights * opts[:routed_scaling_factor]
  end

  # Zeroes out (by setting to the lowest finite value) the experts in all
  # but the `num_groups_per_token` highest scoring groups. Each group is
  # scored with the sum of its two highest expert scores.
  defnp mask_expert_groups(scores, num_groups, num_groups_per_token) do
    grouped = Nx.reshape(scores, {:auto, num_groups, div(Nx.axis_size(scores, -1), num_groups)})

    group_scores =
      grouped
      |> Nx.sort(axis: -1, direction: :desc)
      |> Nx.slice_along_axis(0, 2, axis: -1)
      |> Nx.sum(axes: [-1])

    mask =
      group_scores
      |> top_k_mask(num_groups_per_token)
      |> Nx.new_axis(-1)
      |> Nx.broadcast(Nx.shape(grouped))
      |> Nx.reshape(scores)

    Nx.select(mask, scores, Nx.Constants.min_finite(Nx.type(scores)))
  end

  # Returns a 0/1 mask with ones at the `k` highest values along the last
  # axis. Ties are resolved in favour of lower indices, matching the
  # reference implementations.
  defnp top_k_mask(scores, k) do
    order = Nx.argsort(scores, axis: -1, direction: :desc, stable: true)
    rank = Nx.argsort(order, axis: -1, stable: true)
    Nx.as_type(Nx.less(rank, k), Nx.type(scores))
  end

  @doc """
  Adds a stack of expert feed-forward networks.

  Expects `weights` to be routing weights as returned by `router/2` and
  computes the weighted mixture of all experts.

  ## Options

    * `:num_experts` (required) - the number of experts

    * `:hidden_size` (required) - the dimensionality of the input and
      output

    * `:intermediate_size` (required) - the dimensionality of each expert

    * `:activation` - the activation used within each expert. Defaults
      to `:silu`

    * `:name` - the prefix for layer names

  """
  def experts(hidden_state, weights, opts) do
    validate_required_keys!(opts, [:num_experts, :hidden_size, :intermediate_size])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_experts,
        :hidden_size,
        :intermediate_size,
        activation: :silu,
        variant: :gated,
        use_bias: false,
        clamp_limit: 7.0,
        activation_alpha: 1.702,
        kernel_initializer: :glorot_uniform
      ])

    num_experts = opts[:num_experts]
    hidden_size = opts[:hidden_size]
    intermediate_size = opts[:intermediate_size]

    gate_kernel =
      Axon.param("gate_kernel", fn _, _ -> {num_experts, hidden_size, intermediate_size} end,
        initializer: opts[:kernel_initializer]
      )

    up_kernel =
      Axon.param("up_kernel", fn _, _ -> {num_experts, hidden_size, intermediate_size} end,
        initializer: opts[:kernel_initializer]
      )

    down_kernel =
      Axon.param("down_kernel", fn _, _ -> {num_experts, intermediate_size, hidden_size} end,
        initializer: opts[:kernel_initializer]
      )

    kernels = [gate_kernel, up_kernel, down_kernel]

    {inputs, impl} =
      cond do
        opts[:variant] == :ungated ->
          {[hidden_state, weights, up_kernel, down_kernel], &ungated_experts_impl/5}

        opts[:use_bias] ->
          gate_bias =
            Axon.param("gate_bias", fn _, _ -> {num_experts, intermediate_size} end,
              initializer: :zeros
            )

          up_bias =
            Axon.param("up_bias", fn _, _ -> {num_experts, intermediate_size} end,
              initializer: :zeros
            )

          down_bias =
            Axon.param("down_bias", fn _, _ -> {num_experts, hidden_size} end,
              initializer: :zeros
            )

          {[hidden_state, weights] ++ kernels ++ [gate_bias, up_bias, down_bias],
           &biased_experts_impl/9}

        true ->
          {[hidden_state, weights] ++ kernels, &experts_impl/6}
      end

    Axon.layer(impl, inputs,
      name: opts[:name],
      op_name: :moe_experts,
      activation: opts[:activation],
      variant: opts[:variant],
      clamp_limit: opts[:clamp_limit],
      activation_alpha: opts[:activation_alpha]
    )
  end

  defnp ungated_experts_impl(hidden_state, weights, up_kernel, down_kernel, opts \\ []) do
    opts =
      keyword!(opts, [:activation, :variant, :clamp_limit, :activation_alpha, mode: :inference])

    {batch_size, sequence_length, hidden_size} = Nx.shape(hidden_state)
    tokens = Nx.reshape(hidden_state, {batch_size * sequence_length, hidden_size})
    weights = Nx.reshape(weights, {batch_size * sequence_length, :auto})
    weights = Nx.as_type(weights, Nx.type(tokens))

    intermediate =
      tokens
      |> Nx.dot([1], up_kernel, [1])
      |> apply_activation(opts[:activation])
      |> Nx.multiply(Nx.new_axis(weights, -1))

    output = Nx.dot(intermediate, [1, 2], down_kernel, [0, 1])

    Nx.reshape(output, {batch_size, sequence_length, hidden_size})
  end

  defnp experts_impl(hidden_state, weights, gate_kernel, up_kernel, down_kernel, opts \\ []) do
    biased_experts_impl(
      hidden_state,
      weights,
      gate_kernel,
      up_kernel,
      down_kernel,
      0.0,
      0.0,
      0.0,
      opts
    )
  end

  defnp biased_experts_impl(
          hidden_state,
          weights,
          gate_kernel,
          up_kernel,
          down_kernel,
          gate_bias,
          up_bias,
          down_bias,
          opts \\ []
        ) do
    opts =
      keyword!(opts, [
        :activation,
        :variant,
        :clamp_limit,
        :activation_alpha,
        mode: :inference
      ])

    {batch_size, sequence_length, hidden_size} = Nx.shape(hidden_state)
    tokens = Nx.reshape(hidden_state, {batch_size * sequence_length, hidden_size})
    weights = Nx.reshape(weights, {batch_size * sequence_length, :auto})
    weights = Nx.as_type(weights, Nx.type(tokens))

    gate = Nx.dot(tokens, [1], gate_kernel, [1]) + gate_bias
    up = Nx.dot(tokens, [1], up_kernel, [1]) + up_bias

    intermediate =
      case opts[:variant] do
        :gated ->
          apply_activation(gate, opts[:activation]) * up

        :clamped_glu ->
          limit = opts[:clamp_limit]
          gate = Nx.min(gate, limit)
          up = Nx.clip(up, -limit, limit)
          (up + 1) * (gate * Nx.sigmoid(gate * opts[:activation_alpha]))

        :clamped_gated ->
          limit = opts[:clamp_limit]
          gate = Nx.min(gate, limit)
          up = Nx.clip(up, -limit, limit)
          apply_activation(gate, opts[:activation]) * up
      end

    intermediate = intermediate * Nx.new_axis(weights, -1)

    output =
      Nx.dot(intermediate, [1, 2], down_kernel, [0, 1]) + down_bias_output(weights, down_bias)

    Nx.reshape(output, {batch_size, sequence_length, hidden_size})
  end

  # The output bias of every expert is scaled by its routing weight
  deftransformp down_bias_output(weights, down_bias) do
    if Nx.rank(down_bias) == 2 do
      Nx.dot(weights, [1], down_bias, [0])
    else
      0.0
    end
  end

  deftransformp apply_activation(input, activation) do
    case activation do
      :gelu_approx_tanh -> Layers.gelu_approx_tanh(input)
      :gelu_approx_sigmoid -> Layers.gelu_approx_sigmoid(input)
      :relu_squared -> Layers.relu_squared(input)
      activation -> apply(Axon.Activations, activation, [input])
    end
  end

  @doc """
  Adds a regular (non-gated) feed-forward network.
  """
  def ffn(hidden_state, intermediate_size, output_size, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        activation: :silu,
        use_bias: false,
        kernel_initializer: :glorot_uniform
      ])

    name = opts[:name]

    hidden_state
    |> Axon.dense(intermediate_size,
      kernel_initializer: opts[:kernel_initializer],
      name: join(name, "intermediate"),
      use_bias: opts[:use_bias]
    )
    |> Layers.activation(opts[:activation])
    |> Axon.dense(output_size,
      kernel_initializer: opts[:kernel_initializer],
      name: join(name, "output"),
      use_bias: opts[:use_bias]
    )
  end

  @doc """
  Adds a gated feed-forward network, as used by most recent LLMs.
  """
  def gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    opts =
      Keyword.validate!(opts, [:name, activation: :silu, kernel_initializer: :glorot_uniform])

    name = opts[:name]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        kernel_initializer: opts[:kernel_initializer],
        name: join(name, "intermediate"),
        use_bias: false
      )

    gate =
      Axon.dense(hidden_state, intermediate_size,
        kernel_initializer: opts[:kernel_initializer],
        name: join(name, "gate"),
        use_bias: false
      )

    hidden_state = Axon.multiply(intermediate, Layers.activation(gate, opts[:activation]))

    Axon.dense(hidden_state, output_size,
      kernel_initializer: opts[:kernel_initializer],
      name: join(name, "output"),
      use_bias: false
    )
  end

  @doc """
  Builds parameter mapping entries for a MoE block built with `block/2`.

  `python_layer_name` should point at the Python MoE module, such as
  `"model.layers.{n}.mlp"`.

  The routed expert weights are looked up either as individual per-expert
  linear layers (`mlp.experts.0.gate_proj.weight`), which is what the
  released checkpoints use, or as stacked tensors
  (`mlp.experts.gate_up_proj`), which is what newer versions of
  huggingface/transformers save.
  """
  def params_mapping(bumblebee_layer_name, python_layer_name, opts) do
    opts = Keyword.validate!(opts, [:num_experts, :intermediate_size, shared_expert: false])

    num_experts = Keyword.fetch!(opts, :num_experts)
    intermediate_size = Keyword.fetch!(opts, :intermediate_size)

    experts = join(python_layer_name, "experts")

    expert_refs = fn projection, packed ->
      for idx <- 0..(num_experts - 1) do
        [
          {join(experts, "#{idx}.#{projection}"), "weight"},
          {experts, packed}
        ]
      end
    end

    %{
      join(bumblebee_layer_name, "router") => %{
        "kernel" => {
          [{join(python_layer_name, "gate"), "weight"}],
          fn [kernel] -> Nx.transpose(kernel) end
        },
        "score_correction_bias" => {
          [{join(python_layer_name, "gate"), "e_score_correction_bias"}],
          fn [bias] -> bias end
        }
      },
      join(bumblebee_layer_name, "experts") => %{
        "gate_kernel" => {
          expert_refs.("gate_proj", "gate_up_proj"),
          &stack_expert_kernels(&1, 0, intermediate_size)
        },
        "up_kernel" => {
          expert_refs.("up_proj", "gate_up_proj"),
          &stack_expert_kernels(&1, intermediate_size, intermediate_size)
        },
        "down_kernel" => {
          expert_refs.("down_proj", "down_proj"),
          &stack_expert_kernels(&1, 0, nil)
        }
      }
    }
    |> Map.merge(
      if opts[:shared_expert] do
        shared = join(python_layer_name, "shared_experts")

        %{
          join(bumblebee_layer_name, "shared_expert.gate") => join(shared, "gate_proj"),
          join(bumblebee_layer_name, "shared_expert.intermediate") => join(shared, "up_proj"),
          join(bumblebee_layer_name, "shared_expert.output") => join(shared, "down_proj")
        }
      else
        %{}
      end
    )
  end

  @doc """
  Builds a `{num_experts, in, out}` kernel out of either a list of
  per-expert `{out, in}` kernels, or a list of references to the same
  stacked `{num_experts, out, in}` tensor, in which case a slice is taken
  for each expert.

  When `size` is given, only that many rows are taken from each expert
  kernel, starting at `offset`. This is used for checkpoints that store
  the gate and up projections as a single tensor.
  """
  def stack_expert_kernels(values, offset \\ 0, size \\ nil) do
    values
    |> Enum.with_index()
    |> Enum.map(fn {value, idx} ->
      value =
        case Nx.rank(value) do
          2 -> value
          3 -> value[idx]
        end

      value =
        if size do
          Nx.slice_along_axis(value, offset, size, axis: 0)
        else
          value
        end

      Nx.transpose(value)
    end)
    |> Nx.stack()
  end

  defp validate_required_keys!(opts, keys) do
    case keys -- Keyword.keys(opts) do
      [] -> :ok
      missing -> raise ArgumentError, "missing required options: #{inspect(missing)}"
    end
  end
end
