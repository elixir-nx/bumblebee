defmodule Bumblebee.SentenceTransformers.Pipeline do
  @moduledoc false

  alias Axon.MixedPrecision.Policy
  alias Bumblebee.Layers
  alias Bumblebee.SentenceTransformers.Modules

  defdelegate pooling_layer(hidden_state, attention_mask, config, name), to: Modules
  defdelegate default_pooling(node, attention_mask, name \\ "default_pooling"), to: Modules
  defdelegate build_static_embedding(repository, module, opts), to: Modules
  defdelegate decode_json(path), to: Modules
  defdelegate cast_param(tensor, type), to: Modules
  defdelegate allocate_param(tensor, backend), to: Modules
  defdelegate load_module_config(repository, path), to: Modules

  @doc """
  Builds the pipeline modules on top of base model_info.
  """
  def build(repository, modules, model_info, opts) do
    base_model = model_info.model

    opts =
      opts
      |> Keyword.put_new_lazy(:type, fn -> infer_param_type(model_info.params) end)
      |> Keyword.put_new_lazy(:backend, fn -> infer_param_backend(model_info.params) end)

    with {:ok, modules} <- maybe_fuse_dense(repository, modules, opts[:fuse_dense]) do
      has_weighted_pooling? =
        Enum.any?(modules, fn m ->
          normalize_module_type(m["type"]) == "WeightedLayerPooling"
        end)

      hidden_state = extract_base_hidden_state(base_model, has_weighted_pooling?)

      attention_mask =
        Layers.default Axon.input("attention_mask", optional: true) do
          Layers.default_attention_mask(Axon.input("input_ids"))
        end

      initial_state =
        {hidden_state, model_info.params.data, model_info.params.parameters, false}

      with {:ok, {embedding, params_data, params_parameters, prompt_length_required}} <-
             build_pipeline_modules(repository, modules, initial_state, attention_mask, opts) do
        embedding = maybe_truncate_embedding(embedding, modules, opts)

        final_model =
          Axon.container(%{
            embedding: embedding,
            pooled_state: embedding,
            hidden_state: hidden_state
          })
          |> apply_type(opts[:type])

        final_params =
          if params_parameters do
            %{model_info.params | data: params_data, parameters: params_parameters}
          else
            %{model_info.params | data: params_data}
          end

        {:ok,
         Map.put_new(
           %{model_info | model: final_model, params: final_params},
           :prompt_length_required,
           prompt_length_required
         )}
      end
    end
  end

  def normalize_module_type(type) when is_binary(type) do
    type
    |> String.split(".")
    |> List.last()
  end

  @doc """
  Returns true if the built model expects a `prompt_length` input.
  This happens when a pooling module has `include_prompt: false`.
  """
  def prompt_length_required?(%{prompt_length_required: value}), do: value
  def prompt_length_required?(_), do: false

  defp extract_base_hidden_state(base_model, true) do
    extract_all_hidden_states(base_model) || extract_single_hidden_state(base_model)
  end

  defp extract_base_hidden_state(base_model, false) do
    extract_single_hidden_state(base_model)
  end

  defp extract_single_hidden_state(base_model) do
    Axon.nx(base_model, fn
      %{hidden_state: hidden_state} ->
        hidden_state

      output ->
        raise ArgumentError,
              "expected base model output to contain :hidden_state, but got keys: #{inspect(Map.keys(output))}." <>
                " Ensure you loaded the base encoder model."
    end)
  end

  defp build_pipeline_modules(repository, modules, initial_state, attention_mask, opts) do
    Enum.reduce_while(
      modules,
      {:ok, initial_state},
      fn module, {:ok, {node, params_data, params_parameters, prompt_length_required}} ->
        case Modules.build(
               repository,
               module,
               node,
               attention_mask,
               params_data,
               params_parameters,
               opts
             ) do
          {:ok, node, params_data, params_parameters, requires_prompt_length} ->
            {:cont,
             {:ok,
              {node, params_data, params_parameters,
               prompt_length_required or requires_prompt_length}}}

          {:error, reason} ->
            {:halt, {:error, reason}}
        end
      end
    )
  end

  defp maybe_truncate_embedding(embedding, modules, opts) do
    case opts[:truncate_dim] do
      truncate_dim when is_integer(truncate_dim) and truncate_dim > 0 ->
        truncated =
          Axon.nx(
            embedding,
            fn tensor -> Nx.slice_along_axis(tensor, 0, truncate_dim, axis: -1) end,
            name: "truncate_dim"
          )

        if needs_renorm?(modules, opts) do
          Axon.nx(truncated, &Bumblebee.Utils.Nx.normalize/1, name: "truncate_dim.normalize")
        else
          truncated
        end

      _ ->
        embedding
    end
  end

  defp needs_renorm?(modules, opts) do
    opts[:normalize] == true or
      opts[:normalize_embeddings] == true or
      pipeline_has_normalize?(modules)
  end

  defp pipeline_has_normalize?(modules) do
    Enum.any?(modules, fn m ->
      normalize_module_type(m["type"]) == "Normalize"
    end)
  end

  defp maybe_fuse_dense(repository, modules, true) do
    fuse_dense_modules(repository, modules, [])
  end

  defp maybe_fuse_dense(_repository, modules, _false), do: {:ok, modules}

  defp fuse_dense_modules(repository, [m1, m2 | rest], acc) do
    case try_fuse_dense(repository, m1, m2) do
      {:ok, fused} -> fuse_dense_modules(repository, rest, [fused | acc])
      :error -> fuse_dense_modules(repository, [m2 | rest], [m1 | acc])
    end
  end

  defp fuse_dense_modules(_repository, [other], acc) do
    {:ok, Enum.reverse([other | acc])}
  end

  defp fuse_dense_modules(_repository, [], acc) do
    {:ok, Enum.reverse(acc)}
  end

  defp try_fuse_dense(
         repository,
         %{"type" => t1, "path" => p1},
         %{"type" => t2, "path" => p2} = _m2
       ) do
    if normalize_module_type(t1) == "Dense" and normalize_module_type(t2) == "Dense" do
      with {:ok, c1} <- Modules.load_module_config(repository, p1),
           {:ok, c2} <- Modules.load_module_config(repository, p2),
           true <- can_fuse_dense?(c1, c2) do
        {:ok,
         %{
           "type" => "sentence_transformers.models.FusedDense",
           "path" => "#{p1}_#{p2}",
           "paths" => [p1, p2],
           "out_features" => c2["out_features"]
         }}
      else
        _ -> :error
      end
    else
      :error
    end
  end

  defp try_fuse_dense(_repository, _m1, _m2), do: :error

  defp can_fuse_dense?(c1, c2) do
    c1_linear? = c1["activation_function"] in [nil, "torch.nn.modules.linear.Identity"]
    c2_linear? = c2["activation_function"] in [nil, "torch.nn.modules.linear.Identity"]
    no_bias? = !Map.get(c1, "bias", true) and !Map.get(c2, "bias", true)
    dims_match? = c1["out_features"] == c2["in_features"]

    c1_linear? and c2_linear? and no_bias? and dims_match?
  end

  defp apply_type(model, nil), do: model

  defp apply_type(model, %Policy{} = policy) do
    Axon.MixedPrecision.apply_policy(model, policy)
  end

  defp apply_type(model, type) do
    type = Nx.Type.normalize!(type)
    policy = Axon.MixedPrecision.create_policy(params: type, compute: type, output: type)
    Axon.MixedPrecision.apply_policy(model, policy)
  end

  defp infer_param_type(%Axon.ModelState{data: data}) do
    find_first_tensor(data, &Nx.type/1)
  end

  defp infer_param_type(_), do: nil

  defp infer_param_backend(%Axon.ModelState{data: data}) do
    find_first_tensor(data, fn %Nx.Tensor{data: %backend{}} ->
      if backend == Nx.BinaryBackend, do: nil, else: backend
    end)
  end

  defp infer_param_backend(_), do: nil

  defp find_first_tensor(data, fun) when is_map(data) do
    Enum.find_value(data, fn
      {_key, %Nx.Tensor{} = tensor} -> fun.(tensor)
      {_key, nested} when is_map(nested) -> find_first_tensor(nested, fun)
      _ -> nil
    end)
  end

  defp find_first_tensor(_, _fun), do: nil

  defp extract_all_hidden_states(%Axon{nodes: nodes}) do
    opt_in_node =
      Enum.find_value(nodes, fn {_id, node} ->
        if node.op_name == :global_opt_in and :output_hidden_states in node.global_options do
          node
        end
      end)

    if opt_in_node do
      [raw_id] = opt_in_node.parent
      raw_axon = %Axon{output: raw_id, nodes: nodes}

      Axon.nx(
        raw_axon,
        fn tuple_hs ->
          tuple_hs |> Tuple.to_list() |> Nx.stack(axis: 0)
        end,
        name: "stacked_hidden_states"
      )
    end
  end

  defp extract_all_hidden_states(_), do: nil
end
