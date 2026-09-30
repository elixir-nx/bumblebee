defmodule Bumblebee.HuggingFace.SentenceTransformers do
  @moduledoc false

  alias Bumblebee.Layers

  @doc """
  Loads a SentenceTransformers embedding head and attaches it to the model.
  """
  def load_embedding_head(_repository, repo_files, download_fun, model_info, opts) do
    case repo_files do
      %{"modules.json" => _etag} ->
        with {:ok, modules_path} <- download_fun.("modules.json"),
             {:ok, modules} <- decode_json(modules_path) do
          modules = Enum.sort_by(modules, & &1["idx"])

          case modules do
            [%{"type" => "sentence_transformers.models.Transformer"} | remaining] ->
              build_pipeline(remaining, repo_files, download_fun, model_info, opts)

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
    hidden_state = Axon.nx(base_model, & &1.hidden_state)

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
         {:ok, config} <- decode_json(config_path) do
      node = pooling_layer(node, attention_mask, config, path)
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
         {:ok, weights_file} <- find_weights_file(repo_files, path),
         {:ok, weights_path} <- download_fun.(weights_file) do
      out_features = config["out_features"]
      bias = Map.get(config, "bias", true)
      activation = parse_activation(config["activation_function"])

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
      for {key, mode} <- [
            {"pooling_mode_cls_token", :cls_token},
            {"pooling_mode_mean_tokens", :mean_tokens},
            {"pooling_mode_max_tokens", :max_tokens},
            {"pooling_mode_mean_sqrt_len_tokens", :mean_sqrt_len_tokens},
            {"pooling_mode_lasttoken", :last_token}
          ],
          config[key] == true,
          do: mode

    Axon.layer(
      fn hidden_state, attention_mask, _opts ->
        pool_outputs =
          Enum.map(modes, fn
            :mean_tokens ->
              mask = Nx.new_axis(attention_mask, -1)
              sum_embeddings = Nx.sum(Nx.multiply(hidden_state, mask), axes: [1])
              sum_mask = Nx.sum(mask, axes: [1]) |> Nx.max(1.0e-9)
              Nx.divide(sum_embeddings, sum_mask)

            :cls_token ->
              hidden_state[[.., 0, ..]]

            :max_tokens ->
              mask = Nx.new_axis(attention_mask, -1)

              hidden_state
              |> Nx.select(mask, Nx.Constants.min_finite(hidden_state))
              |> Nx.reduce_max(axes: [1])

            :mean_sqrt_len_tokens ->
              mask = Nx.new_axis(attention_mask, -1)
              sum_embeddings = Nx.sum(Nx.multiply(hidden_state, mask), axes: [1])
              sum_mask = Nx.sum(mask, axes: [1]) |> Nx.max(1.0e-9)
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
  end

  defp parse_activation("torch.nn.modules.linear.Identity"), do: nil
  defp parse_activation("torch.nn.modules.activation.Tanh"), do: :tanh
  defp parse_activation("torch.nn.modules.activation.ReLU"), do: :relu
  defp parse_activation("torch.nn.modules.activation.GELU"), do: :gelu
  defp parse_activation("torch.nn.modules.activation.SiLU"), do: :silu
  defp parse_activation(nil), do: nil
  defp parse_activation(other), do: raise("unsupported activation function #{inspect(other)}")

  defp maybe_activation(node, nil), do: node
  defp maybe_activation(node, activation), do: Axon.activation(node, activation)

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
