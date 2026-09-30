defmodule Bumblebee.HuggingFace.SentenceTransformers do
  @moduledoc false

  alias Bumblebee.Layers

  @doc """
  Loads a SentenceTransformers embedding head and attaches it to the model.
  """
  def load_embedding_head(_repository, repo_files, download_fun, model_info, opts) do
    opts =
      opts
      |> Keyword.put_new_lazy(:type, fn -> infer_param_type(model_info.params) end)
      |> Keyword.put_new_lazy(:backend, fn -> infer_param_backend(model_info.params) end)

    case repo_files do
      %{"modules.json" => _etag} ->
        with {:ok, modules_path} <- download_fun.("modules.json"),
             {:ok, modules} <- decode_json(modules_path) do
          modules = Enum.sort_by(modules, &(&1["idx"] || 0))

          case modules do
            [%{"type" => "sentence_transformers.models.Transformer"} | remaining] ->
              with {:ok, remaining} <-
                     maybe_fuse_dense(remaining, opts[:fuse_dense], repo_files, download_fun) do
                build_pipeline(remaining, repo_files, download_fun, model_info, opts)
              end

            _other ->
              {:error,
               "expected the first module in modules.json to be sentence_transformers.models.Transformer"}
          end
        end

      _ ->
        {:error, "could not find modules.json in the repository"}
    end
  end

  defp build_pipeline(modules, repo_files, download_fun, model_info, opts) do
    base_model = model_info.model

    hidden_state =
      Axon.nx(base_model, fn
        %{hidden_state: hidden_state} ->
          hidden_state

        output ->
          raise ArgumentError,
                "expected model output to contain :hidden_state, but got keys: #{inspect(Map.keys(output))}." <>
                  " If the model defaults to a language modeling architecture, please specify" <>
                  " `architecture: :base` when calling `Bumblebee.load_model/2`."
      end)

    attention_mask =
      Layers.default Axon.input("attention_mask", optional: true) do
        Layers.default_attention_mask(Axon.input("input_ids"))
      end

    initial_state = {hidden_state, model_info.params.data}

    result =
      Enum.reduce_while(modules, {:ok, initial_state}, fn module, {:ok, {node, params_data}} ->
        case build_module(
               module,
               node,
               attention_mask,
               params_data,
               repo_files,
               download_fun,
               opts
             ) do
          {:ok, node, params_data} ->
            {:cont, {:ok, {node, params_data}}}

          {:error, reason} ->
            {:halt, {:error, reason}}
        end
      end)

    with {:ok, {embedding, params_data}} <- result do
      final_model = Axon.container(%{embedding: embedding, hidden_state: hidden_state})
      final_model = apply_type(final_model, opts[:type])
      final_params = %{model_info.params | data: params_data}

      {:ok, %{model_info | model: final_model, params: final_params}}
    end
  end

  defp build_module(
         %{"type" => "sentence_transformers.models.Pooling", "path" => path},
         node,
         attention_mask,
         params_data,
         _repo_files,
         download_fun,
         _opts
       ) do
    config_file = Path.join(path, "config.json")

    with {:ok, config_path} <- download_fun.(config_file),
         {:ok, config} <- decode_json(config_path),
         {:ok, node} <- pooling_layer(node, attention_mask, config, path) do
      {:ok, node, params_data}
    end
  end

  defp build_module(
         %{"type" => "sentence_transformers.models.Dense", "path" => path},
         node,
         _attention_mask,
         params_data,
         repo_files,
         download_fun,
         opts
       ) do
    config_file = Path.join(path, "config.json")

    with {:ok, config_path} <- download_fun.(config_file),
         {:ok, config} <- decode_json(config_path),
         {:ok, activation} <- parse_activation(config["activation_function"], path),
         {:ok, weights_file} <- find_weights_file(repo_files, path),
         {:ok, weights_path} <- download_fun.(weights_file) do
      out_features = config["out_features"]
      bias = Map.get(config, "bias", true)

      layer_name = path

      node =
        node
        |> Axon.dense(out_features, use_bias: bias, name: layer_name)
        |> maybe_activation(activation)

      tensors = load_tensors(weights_file, weights_path, opts)

      kernel =
        tensors["linear.weight"]
        |> Nx.to_tensor()
        |> Nx.transpose()
        |> cast_param(opts[:type])
        |> allocate_param(opts[:backend])

      layer_params =
        if bias do
          bias_tensor =
            tensors["linear.bias"]
            |> Nx.to_tensor()
            |> cast_param(opts[:type])
            |> allocate_param(opts[:backend])

          %{"kernel" => kernel, "bias" => bias_tensor}
        else
          %{"kernel" => kernel}
        end

      params_data = Map.put(params_data, layer_name, layer_params)
      {:ok, node, params_data}
    end
  end

  defp build_module(
         %{
           "type" => "sentence_transformers.models.FusedDense",
           "path" => layer_name,
           "paths" => [p1, p2],
           "out_features" => out_features
         },
         node,
         _attention_mask,
         params_data,
         repo_files,
         download_fun,
         opts
       ) do
    with {:ok, w1_file} <- find_weights_file(repo_files, p1),
         {:ok, w1_path} <- download_fun.(w1_file),
         {:ok, w2_file} <- find_weights_file(repo_files, p2),
         {:ok, w2_path} <- download_fun.(w2_file) do
      t1 = load_tensors(w1_file, w1_path, opts)
      t2 = load_tensors(w2_file, w2_path, opts)

      k1 = t1["linear.weight"] |> Nx.to_tensor() |> Nx.transpose()
      k2 = t2["linear.weight"] |> Nx.to_tensor() |> Nx.transpose()

      k_fused =
        Nx.dot(k1, k2)
        |> cast_param(opts[:type])
        |> allocate_param(opts[:backend])

      node = Axon.dense(node, out_features, use_bias: false, name: layer_name)
      params_data = Map.put(params_data, layer_name, %{"kernel" => k_fused})

      {:ok, node, params_data}
    end
  end

  defp build_module(
         %{"type" => "sentence_transformers.models.Normalize", "path" => path},
         node,
         _attention_mask,
         params_data,
         _repo_files,
         _download_fun,
         _opts
       ) do
    node = Axon.nx(node, &Bumblebee.Utils.Nx.normalize/1, name: "#{path}.normalize")
    {:ok, node, params_data}
  end

  defp build_module(
         %{"type" => "sentence_transformers.models.Dropout"},
         node,
         _attention_mask,
         params_data,
         _repo_files,
         _download_fun,
         _opts
       ) do
    {:ok, node, params_data}
  end

  defp build_module(%{"type" => type}, _node, _mask, _params, _repo_files, _download_fun, _opts) do
    {:error, "unsupported SentenceTransformers module #{inspect(type)}"}
  end

  defp pooling_layer(hidden_state, attention_mask, config, name) do
    modes =
      for {keys, mode} <- [
            {["pooling_mode_cls_token"], :cls_token},
            {["pooling_mode_max_tokens"], :max_tokens},
            {["pooling_mode_mean_tokens"], :mean_tokens},
            {["pooling_mode_mean_sqrt_len_tokens"], :mean_sqrt_len_tokens},
            {["pooling_mode_lasttoken", "pooling_mode_last_token"], :last_token}
          ],
          Enum.any?(keys, &(config[&1] == true)),
          do: mode

    if modes == [] do
      {:error, "no supported pooling mode found in #{name}"}
    else
      layer =
        Axon.layer(
          fn hidden_state, attention_mask, _opts ->
            type = Nx.type(hidden_state)

            eps =
              case type do
                {:f, 16} -> Nx.tensor(1.0e-4, type: type)
                {:bf, 16} -> Nx.tensor(1.0e-4, type: type)
                _ -> Nx.tensor(1.0e-9, type: type)
              end

            pool_outputs =
              Enum.map(modes, fn
                :mean_tokens ->
                  mask = attention_mask |> Nx.as_type(type) |> Nx.new_axis(-1)
                  sum_embeddings = Nx.sum(Nx.multiply(hidden_state, mask), axes: [1])
                  sum_mask = Nx.sum(mask, axes: [1]) |> Nx.max(eps)
                  Nx.divide(sum_embeddings, sum_mask)

                :cls_token ->
                  hidden_state[[.., 0, ..]]

                :max_tokens ->
                  mask = Nx.new_axis(attention_mask, -1)
                  pred = Nx.broadcast(Nx.not_equal(mask, 0), Nx.shape(hidden_state))

                  pred
                  |> Nx.select(hidden_state, Nx.Constants.min_finite(type))
                  |> Nx.reduce_max(axes: [1])

                :mean_sqrt_len_tokens ->
                  mask = attention_mask |> Nx.as_type(type) |> Nx.new_axis(-1)
                  sum_embeddings = Nx.sum(Nx.multiply(hidden_state, mask), axes: [1])
                  sum_mask = Nx.sum(mask, axes: [1]) |> Nx.max(eps)
                  Nx.divide(sum_embeddings, Nx.sqrt(sum_mask))

                :last_token ->
                  lengths =
                    attention_mask
                    |> Nx.sum(axes: [1])
                    |> Nx.subtract(1)
                    |> Nx.as_type({:s, 64})

                  Bumblebee.Utils.Nx.batched_take(hidden_state, lengths)
              end)

            case pool_outputs do
              [single] -> single
              multiple -> Nx.concatenate(multiple, axis: -1)
            end
          end,
          [hidden_state, attention_mask],
          name: name
        )

      {:ok, layer}
    end
  end

  defp parse_activation("torch.nn.modules.linear.Identity", _path), do: {:ok, nil}
  defp parse_activation("torch.nn.modules.activation.Tanh", _path), do: {:ok, :tanh}
  defp parse_activation("torch.nn.modules.activation.ReLU", _path), do: {:ok, :relu}
  defp parse_activation("torch.nn.modules.activation.GELU", _path), do: {:ok, :gelu}
  defp parse_activation("torch.nn.modules.activation.SiLU", _path), do: {:ok, :silu}
  defp parse_activation(nil, _path), do: {:ok, nil}

  defp parse_activation(other, path) do
    {:error, "unsupported activation function #{inspect(other)} in #{path}"}
  end

  defp maybe_activation(node, nil), do: node
  defp maybe_activation(node, activation), do: Axon.activation(node, activation)

  defp maybe_fuse_dense(modules, true, repo_files, download_fun) do
    fuse_dense_modules(modules, repo_files, download_fun, [])
  end

  defp maybe_fuse_dense(modules, _false, _repo_files, _download_fun), do: {:ok, modules}

  defp fuse_dense_modules(
         [
           %{"type" => "sentence_transformers.models.Dense", "path" => p1} = m1,
           %{"type" => "sentence_transformers.models.Dense", "path" => p2} = m2 | rest
         ],
         repo_files,
         download_fun,
         acc
       ) do
    with {:ok, c1} <- load_module_config(p1, download_fun),
         {:ok, c2} <- load_module_config(p2, download_fun),
         true <- can_fuse_dense?(c1, c2) do
      fused_module = %{
        "type" => "sentence_transformers.models.FusedDense",
        "path" => "#{p1}_#{p2}",
        "paths" => [p1, p2],
        "out_features" => c2["out_features"]
      }

      fuse_dense_modules(rest, repo_files, download_fun, [fused_module | acc])
    else
      _ ->
        fuse_dense_modules([m2 | rest], repo_files, download_fun, [m1 | acc])
    end
  end

  defp fuse_dense_modules([other | rest], repo_files, download_fun, acc) do
    fuse_dense_modules(rest, repo_files, download_fun, [other | acc])
  end

  defp fuse_dense_modules([], _repo_files, _download_fun, acc) do
    {:ok, Enum.reverse(acc)}
  end

  defp load_module_config(path, download_fun) do
    config_file = Path.join(path, "config.json")

    with {:ok, config_path} <- download_fun.(config_file),
         {:ok, config} <- decode_json(config_path) do
      {:ok, config}
    end
  end

  defp can_fuse_dense?(c1, c2) do
    c1_linear? = c1["activation_function"] in [nil, "torch.nn.modules.linear.Identity"]
    c2_linear? = c2["activation_function"] in [nil, "torch.nn.modules.linear.Identity"]
    no_bias? = !Map.get(c1, "bias", true) and !Map.get(c2, "bias", true)
    dims_match? = c1["out_features"] == c2["in_features"]

    c1_linear? and c2_linear? and no_bias? and dims_match?
  end

  defp find_weights_file(repo_files, dir) do
    safetensors = Path.join(dir, "model.safetensors")
    pytorch_bin = Path.join(dir, "pytorch_model.bin")

    cond do
      Map.has_key?(repo_files, safetensors) ->
        {:ok, safetensors}

      Map.has_key?(repo_files, pytorch_bin) ->
        {:ok, pytorch_bin}

      true ->
        {:error, "could not find parameters file in #{dir}"}
    end
  end

  defp load_tensors(weights_file, weights_path, opts) do
    case Path.extname(weights_file) do
      ".safetensors" ->
        reader = opts[:safetensors_reader] || (&Safetensors.read!(&1, lazy: true))
        reader.(weights_path)

      _ ->
        Bumblebee.Conversion.PyTorchLoader.load!(weights_path)
    end
  end

  defp cast_param(tensor, nil), do: tensor

  defp cast_param(tensor, %Axon.MixedPrecision.Policy{params: type}) do
    Nx.as_type(tensor, type)
  end

  defp cast_param(tensor, type) do
    type = Nx.Type.normalize!(type)
    Nx.as_type(tensor, type)
  end

  defp allocate_param(tensor, nil), do: tensor

  defp allocate_param(tensor, backend) do
    Nx.with_default_backend(backend, fn -> Nx.backend_copy(tensor) end)
  end

  defp apply_type(model, nil), do: model

  defp apply_type(model, %Axon.MixedPrecision.Policy{} = policy) do
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

  defp decode_json(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} -> {:ok, data}
          _ -> {:error, "failed to parse #{path} as JSON"}
        end

      {:error, reason} ->
        {:error, "failed to read #{path}, reason: #{:file.format_error(reason)}"}
    end
  end
end
