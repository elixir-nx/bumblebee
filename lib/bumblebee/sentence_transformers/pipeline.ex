defmodule Bumblebee.SentenceTransformers.Pipeline do
  @moduledoc false

  alias Axon.MixedPrecision.Policy
  alias Bumblebee.Layers
  alias Bumblebee.SentenceTransformers.Downloader

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

      extracted_hidden_states =
        if has_weighted_pooling?, do: extract_all_hidden_states(base_model)

      hidden_state =
        if extracted_hidden_states do
          extracted_hidden_states
        else
          Axon.nx(base_model, fn
            %{hidden_state: hidden_state} ->
              hidden_state

            output ->
              raise ArgumentError,
                    "expected base model output to contain :hidden_state, but got keys: #{inspect(Map.keys(output))}." <>
                      " Ensure you loaded the base encoder model."
          end)
        end

      attention_mask =
        Layers.default Axon.input("attention_mask", optional: true) do
          Layers.default_attention_mask(Axon.input("input_ids"))
        end

      initial_state =
        {hidden_state, model_info.params.data, model_info.params.parameters, false}

      result =
        Enum.reduce_while(
          modules,
          {:ok, initial_state},
          fn module, {:ok, {node, params_data, params_parameters, prompt_length_required}} ->
            case build_module(
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

      with {:ok, {embedding, params_data, params_parameters, prompt_length_required}} <- result do
        embedding =
          case opts[:truncate_dim] do
            truncate_dim when is_integer(truncate_dim) and truncate_dim > 0 ->
              truncated =
                Axon.nx(
                  embedding,
                  fn tensor -> Nx.slice_along_axis(tensor, 0, truncate_dim, axis: -1) end,
                  name: "truncate_dim"
                )

              needs_renorm? =
                opts[:normalize] == true or
                  opts[:normalize_embeddings] == true or
                  pipeline_has_normalize?(modules)

              if needs_renorm? do
                Axon.nx(truncated, &Bumblebee.Utils.Nx.normalize/1,
                  name: "truncate_dim.normalize"
                )
              else
                truncated
              end

            _ ->
              embedding
          end

        final_model =
          Axon.container(%{
            embedding: embedding,
            pooled_state: embedding,
            hidden_state: hidden_state
          })

        final_model = apply_type(final_model, opts[:type])

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

  defp build_module(
         repository,
         %{"path" => path} = module,
         node,
         attention_mask,
         params_data,
         params_parameters,
         opts
       ) do
    case normalize_module_type(module["type"]) do
      "Pooling" ->
        config_result =
          cond do
            config = module["config"] ->
              {:ok, config}

            path == "default_pooling" ->
              mode = module["pooling_mode"] || "mean"
              {:ok, %{"pooling_mode" => mode}}

            true ->
              with {:ok, config_path} <-
                     Downloader.download_file(repository, Path.join(path, "config.json")) do
                decode_json(config_path)
              end
          end

        with {:ok, config} <- config_result,
             {:ok, node} <- pooling_layer(node, attention_mask, config, path) do
          requires_prompt_length = Map.get(config, "include_prompt", true) == false
          {:ok, node, params_data, params_parameters, requires_prompt_length}
        end

      "Dense" ->
        with {:ok, config_path} <-
               Downloader.download_file(repository, Path.join(path, "config.json")),
             {:ok, config} <- decode_json(config_path),
             {:ok, activation} <- parse_activation(config["activation_function"], path),
             {:ok, weights_file, weights_path} <- find_weights_file(repository, path) do
          in_features = config["in_features"]
          out_features = config["out_features"]
          bias = Map.get(config, "bias", true)
          use_residual = Map.get(config, "use_residual", false)
          layer_name = path

          out_node =
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

          params_parameters =
            if params_parameters do
              Map.put(params_parameters, layer_name, Map.keys(layer_params))
            end

          {node, params_data, params_parameters} =
            if use_residual do
              if in_features == out_features do
                {Axon.add(out_node, node, name: "#{layer_name}.add_residual"), params_data,
                 params_parameters}
              else
                res_name = "#{layer_name}.residual"
                res_node = Axon.dense(node, out_features, use_bias: false, name: res_name)

                res_kernel =
                  tensors["residual.weight"]
                  |> Nx.to_tensor()
                  |> Nx.transpose()
                  |> cast_param(opts[:type])
                  |> allocate_param(opts[:backend])

                res_params = %{"kernel" => res_kernel}
                params_data = Map.put(params_data, res_name, res_params)

                params_parameters =
                  if params_parameters do
                    Map.put(params_parameters, res_name, ["kernel"])
                  end

                {Axon.add(out_node, res_node, name: "#{layer_name}.add_residual"), params_data,
                 params_parameters}
              end
            else
              {out_node, params_data, params_parameters}
            end

          {:ok, node, params_data, params_parameters, false}
        end

      "FusedDense" ->
        [p1, p2] = module["paths"]
        layer_name = module["path"]
        out_features = module["out_features"]

        with {:ok, w1_file, w1_path} <- find_weights_file(repository, p1),
             {:ok, w2_file, w2_path} <- find_weights_file(repository, p2) do
          t1 = load_tensors(w1_file, w1_path, opts)
          t2 = load_tensors(w2_file, w2_path, opts)

          k1 = t1["linear.weight"] |> Nx.to_tensor() |> Nx.transpose()
          k2 = t2["linear.weight"] |> Nx.to_tensor() |> Nx.transpose()

          k_fused =
            k1
            |> Nx.dot(k2)
            |> cast_param(opts[:type])
            |> allocate_param(opts[:backend])

          node = Axon.dense(node, out_features, use_bias: false, name: layer_name)
          params_data = Map.put(params_data, layer_name, %{"kernel" => k_fused})

          params_parameters =
            if params_parameters do
              Map.put(params_parameters, layer_name, ["kernel"])
            end

          {:ok, node, params_data, params_parameters, false}
        end

      "Normalize" ->
        node = Axon.nx(node, &Bumblebee.Utils.Nx.normalize/1, name: "#{path}.normalize")
        {:ok, node, params_data, params_parameters, false}

      "Dropout" ->
        {:ok, node, params_data, params_parameters, false}

      "LayerNorm" ->
        with {:ok, config_path} <-
               Downloader.download_file(repository, Path.join(path, "config.json")),
             {:ok, config} <- decode_json(config_path),
             {:ok, weights_file, weights_path} <- find_weights_file(repository, path) do
          eps = config["eps"] || 1.0e-5
          layer_name = path

          node = Axon.layer_norm(node, epsilon: eps, name: layer_name)
          tensors = load_tensors(weights_file, weights_path, opts)

          gamma =
            (tensors["weight"] || tensors["gamma"] || tensors["norm.weight"])
            |> Nx.to_tensor()
            |> cast_param(opts[:type])
            |> allocate_param(opts[:backend])

          beta =
            (tensors["bias"] || tensors["beta"] || tensors["norm.bias"])
            |> Nx.to_tensor()
            |> cast_param(opts[:type])
            |> allocate_param(opts[:backend])

          layer_params = %{"gamma" => gamma, "beta" => beta}
          params_data = Map.put(params_data, layer_name, layer_params)

          params_parameters =
            if params_parameters do
              Map.put(params_parameters, layer_name, ["gamma", "beta"])
            end

          {:ok, node, params_data, params_parameters, false}
        end

      "WeightedLayerPooling" ->
        with {:ok, config_path} <-
               Downloader.download_file(repository, Path.join(path, "config.json")),
             {:ok, config} <- decode_json(config_path) do
          layer_start = config["layer_start"] || 4
          num_hidden_layers = config["num_hidden_layers"] || 12
          num_weights = max(num_hidden_layers + 1 - layer_start, 1)
          layer_name = path

          {_weights, params_data, params_parameters} =
            case find_weights_file(repository, path) do
              {:ok, weights_file, weights_path} ->
                tensors = load_tensors(weights_file, weights_path, opts)

                w =
                  (tensors["layer_weights"] || tensors["weight"] ||
                     Nx.broadcast(1.0, {num_weights}))
                  |> Nx.to_tensor()
                  |> cast_param(opts[:type])
                  |> allocate_param(opts[:backend])

                params_data = Map.put(params_data, layer_name, %{"layer_weights" => w})

                params_parameters =
                  if params_parameters do
                    Map.put(params_parameters, layer_name, ["layer_weights"])
                  end

                {w, params_data, params_parameters}

              _ ->
                w =
                  1.0
                  |> Nx.broadcast({num_weights})
                  |> cast_param(opts[:type])
                  |> allocate_param(opts[:backend])

                params_data = Map.put(params_data, layer_name, %{"layer_weights" => w})

                params_parameters =
                  if params_parameters do
                    Map.put(params_parameters, layer_name, ["layer_weights"])
                  end

                {w, params_data, params_parameters}
            end

          param_layer = Axon.param("layer_weights", {num_weights}, initializer: :ones)

          weighted_node =
            Axon.layer(
              fn stacked_layers, layer_weights, _opts ->
                total_layers = Nx.axis_size(stacked_layers, 0)

                selected =
                  Nx.slice_along_axis(
                    stacked_layers,
                    layer_start,
                    total_layers - layer_start,
                    axis: 0
                  )

                num_selected = total_layers - layer_start
                weights_reshaped = Nx.reshape(layer_weights, {num_selected, 1, 1, 1})
                weighted = Nx.multiply(weights_reshaped, selected)
                sum = Nx.sum(weighted, axes: [0])
                total_weight = Nx.sum(layer_weights)
                Nx.divide(sum, total_weight)
              end,
              [node, param_layer],
              name: layer_name
            )

          {:ok, weighted_node, params_data, params_parameters, false}
        end

      other ->
        {:error, "unsupported SentenceTransformers module #{inspect(other)}"}
    end
  end

  def default_pooling(node, attention_mask, name \\ "default_pooling") do
    config = %{"pooling_mode_mean_tokens" => true}
    pooling_layer(node, attention_mask, config, name)
  end

  def pooling_layer(hidden_state, attention_mask, config, name) do
    include_prompt = Map.get(config, "include_prompt", true)

    modes_from_flags =
      for {keys, mode} <- [
            {["pooling_mode_cls_token"], :cls_token},
            {["pooling_mode_max_tokens"], :max_tokens},
            {["pooling_mode_mean_tokens"], :mean_tokens},
            {["pooling_mode_mean_sqrt_len_tokens"], :mean_sqrt_len_tokens},
            {["pooling_mode_weightedmean_tokens"], :weightedmean_tokens},
            {["pooling_mode_lasttoken", "pooling_mode_last_token"], :last_token}
          ],
          Enum.any?(keys, &(config[&1] == true)),
          do: mode

    modes =
      case {modes_from_flags, config["pooling_mode"]} do
        {[], "mean"} -> [:mean_tokens]
        {[], "cls"} -> [:cls_token]
        {[], "max"} -> [:max_tokens]
        {[], "mean_sqrt_len_tokens"} -> [:mean_sqrt_len_tokens]
        {[], "weightedmean"} -> [:weightedmean_tokens]
        {[], "last"} -> [:last_token]
        {[], "lasttoken"} -> [:last_token]
        {modes, _} -> modes
      end

    if modes == [] do
      {:error, "no supported pooling mode found in #{name}"}
    else
      layer = build_pooling_layer(hidden_state, attention_mask, modes, include_prompt, name)
      {:ok, layer}
    end
  end

  defp build_pooling_layer(hidden_state, attention_mask, modes, true, name) do
    Axon.layer(
      fn hidden_state, attention_mask, _opts ->
        apply_pooling_modes(hidden_state, attention_mask, modes)
      end,
      [hidden_state, attention_mask],
      name: name
    )
  end

  defp build_pooling_layer(hidden_state, attention_mask, modes, false, name) do
    prompt_length = Axon.input("prompt_length", optional: true)

    Axon.layer(
      fn hidden_state, attention_mask, prompt_length, _opts ->
        attention_mask =
          case prompt_length do
            %Axon.None{} -> attention_mask
            _ -> exclude_prompt_from_mask(attention_mask, prompt_length)
          end

        apply_pooling_modes(hidden_state, attention_mask, modes)
      end,
      [hidden_state, attention_mask, prompt_length],
      name: name
    )
  end

  defp apply_pooling_modes(hidden_state, attention_mask, modes) do
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
          sum_mask = mask |> Nx.sum(axes: [1]) |> Nx.max(eps)
          Nx.divide(sum_embeddings, sum_mask)

        :cls_token ->
          first_indices = Nx.argmax(attention_mask, axis: 1)
          Bumblebee.Utils.Nx.batched_take(hidden_state, first_indices)

        :max_tokens ->
          mask = Nx.new_axis(attention_mask, -1)
          pred = Nx.broadcast(Nx.not_equal(mask, 0), Nx.shape(hidden_state))

          pred
          |> Nx.select(hidden_state, Nx.Constants.neg_infinity(type))
          |> Nx.reduce_max(axes: [1])

        :mean_sqrt_len_tokens ->
          mask = attention_mask |> Nx.as_type(type) |> Nx.new_axis(-1)
          sum_embeddings = Nx.sum(Nx.multiply(hidden_state, mask), axes: [1])
          sum_mask = mask |> Nx.sum(axes: [1]) |> Nx.max(eps)
          Nx.divide(sum_embeddings, Nx.sqrt(sum_mask))

        :weightedmean_tokens ->
          mask = attention_mask |> Nx.as_type(type) |> Nx.new_axis(-1)
          seq_len = Nx.axis_size(hidden_state, 1)

          weights =
            {1, seq_len, 1}
            |> Nx.iota(type: type)
            |> Nx.add(1)

          weighted_mask = Nx.multiply(mask, weights)
          sum_embeddings = Nx.sum(Nx.multiply(hidden_state, weighted_mask), axes: [1])
          sum_mask = weighted_mask |> Nx.sum(axes: [1]) |> Nx.max(eps)
          Nx.divide(sum_embeddings, sum_mask)

        :last_token ->
          lengths =
            attention_mask
            |> Nx.not_equal(0)
            |> Nx.select(
              Nx.iota(Nx.shape(attention_mask), axis: 1),
              0
            )
            |> Nx.reduce_max(axes: [1])
            |> Nx.as_type({:s, 64})

          selected = Bumblebee.Utils.Nx.batched_take(hidden_state, lengths)
          valid = attention_mask |> Nx.not_equal(0) |> Nx.any(axes: [1]) |> Nx.new_axis(-1)
          Nx.select(Nx.broadcast(valid, Nx.shape(selected)), selected, 0)
      end)

    case pool_outputs do
      [single] -> single
      multiple -> Nx.concatenate(multiple, axis: -1)
    end
  end

  defp exclude_prompt_from_mask(attention_mask, prompt_length) do
    type = Nx.type(attention_mask)
    seq_len = Nx.axis_size(attention_mask, 1)
    batch_size = Nx.axis_size(attention_mask, 0)

    prompt_length =
      case Nx.rank(prompt_length) do
        0 -> Nx.broadcast(Nx.as_type(prompt_length, type), {batch_size, 1})
        1 -> prompt_length |> Nx.as_type(type) |> Nx.reshape({batch_size, 1})
        2 -> Nx.as_type(prompt_length, type)
      end

    positions = Nx.iota({1, seq_len}, type: type)

    pad_lengths =
      attention_mask
      |> Nx.as_type({:s, 64})
      |> Nx.argmax(axis: 1)
      |> Nx.as_type(type)
      |> Nx.reshape({batch_size, 1})

    has_left_pad = Nx.any(Nx.not_equal(pad_lengths, 0))

    threshold_right = prompt_length
    threshold_left = Nx.add(pad_lengths, prompt_length)

    threshold =
      Nx.select(Nx.broadcast(has_left_pad, {batch_size, 1}), threshold_left, threshold_right)

    keep_mask = Nx.greater_equal(positions, threshold)

    Nx.multiply(attention_mask, Nx.as_type(keep_mask, type))
  end

  defp parse_activation("torch.nn.modules.linear.Identity", _path), do: {:ok, nil}
  defp parse_activation("torch.nn.modules.activation.Tanh", _path), do: {:ok, :tanh}
  defp parse_activation("torch.nn.modules.activation.ReLU", _path), do: {:ok, :relu}
  defp parse_activation("torch.nn.modules.activation.GELU", _path), do: {:ok, :gelu}
  defp parse_activation("torch.nn.modules.activation.SiLU", _path), do: {:ok, :silu}
  defp parse_activation("torch.nn.modules.activation.Sigmoid", _path), do: {:ok, :sigmoid}
  defp parse_activation("torch.nn.modules.activation.Mish", _path), do: {:ok, :mish}
  defp parse_activation("torch.nn.modules.activation.LeakyReLU", _path), do: {:ok, :leaky_relu}
  defp parse_activation("transformers.activations.GELUActivation", _path), do: {:ok, :gelu}
  defp parse_activation("transformers.activations.FastGELUActivation", _path), do: {:ok, :gelu}
  defp parse_activation("transformers.activations.NewGELUActivation", _path), do: {:ok, :gelu}
  defp parse_activation("transformers.activations.SiLUActivation", _path), do: {:ok, :silu}
  defp parse_activation(nil, _path), do: {:ok, nil}

  defp parse_activation(other, path) do
    {:error, "unsupported activation function #{inspect(other)} in #{path}"}
  end

  defp pipeline_has_normalize?(modules) do
    Enum.any?(modules, fn m ->
      normalize_module_type(m["type"]) == "Normalize"
    end)
  end

  defp maybe_activation(node, nil), do: node
  defp maybe_activation(node, activation), do: Axon.activation(node, activation)

  defp maybe_fuse_dense(repository, modules, true) do
    fuse_dense_modules(repository, modules, [])
  end

  defp maybe_fuse_dense(_repository, modules, _false), do: {:ok, modules}

  defp fuse_dense_modules(
         repository,
         [%{"type" => t1, "path" => p1} = m1, %{"type" => t2, "path" => p2} = m2 | rest],
         acc
       ) do
    if normalize_module_type(t1) == "Dense" and normalize_module_type(t2) == "Dense" do
      with {:ok, c1} <- load_module_config(repository, p1),
           {:ok, c2} <- load_module_config(repository, p2),
           true <- can_fuse_dense?(c1, c2) do
        fused_module = %{
          "type" => "sentence_transformers.models.FusedDense",
          "path" => "#{p1}_#{p2}",
          "paths" => [p1, p2],
          "out_features" => c2["out_features"]
        }

        fuse_dense_modules(repository, rest, [fused_module | acc])
      else
        _ ->
          fuse_dense_modules(repository, [m2 | rest], [m1 | acc])
      end
    else
      fuse_dense_modules(repository, [m2 | rest], [m1 | acc])
    end
  end

  defp fuse_dense_modules(repository, [other | rest], acc) do
    fuse_dense_modules(repository, rest, [other | acc])
  end

  defp fuse_dense_modules(_repository, [], acc) do
    {:ok, Enum.reverse(acc)}
  end

  defp load_module_config(repository, path) do
    with {:ok, config_path} <-
           Downloader.download_file(repository, Path.join(path, "config.json")) do
      decode_json(config_path)
    end
  end

  defp can_fuse_dense?(c1, c2) do
    c1_linear? = c1["activation_function"] in [nil, "torch.nn.modules.linear.Identity"]
    c2_linear? = c2["activation_function"] in [nil, "torch.nn.modules.linear.Identity"]
    no_bias? = !Map.get(c1, "bias", true) and !Map.get(c2, "bias", true)
    dims_match? = c1["out_features"] == c2["in_features"]

    c1_linear? and c2_linear? and no_bias? and dims_match?
  end

  @weights_filenames ["model.safetensors", "pytorch_model.bin"]

  defp find_weights_file(repository, dir) do
    Enum.find_value(@weights_filenames, fn filename ->
      path = Path.join(dir, filename)

      case Downloader.download_file(repository, path) do
        {:ok, downloaded_path} -> {:ok, path, downloaded_path}
        _ -> nil
      end
    end) || {:error, "could not find parameters file in #{dir}"}
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

  @doc false
  def cast_param(tensor, nil), do: tensor

  def cast_param(tensor, %Policy{params: type}) do
    Nx.as_type(tensor, type)
  end

  def cast_param(tensor, type) do
    type = Nx.Type.normalize!(type)
    Nx.as_type(tensor, type)
  end

  @doc false
  def allocate_param(tensor, nil), do: tensor

  def allocate_param(tensor, backend) do
    Nx.with_default_backend(backend, fn -> Nx.backend_copy(tensor) end)
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

  def decode_json(path) do
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

  @doc """
  Builds a base model from a StaticEmbedding module (e.g. Model2Vec).
  """
  def build_static_embedding(repository, %{"path" => path}, opts) do
    with {:ok, weights_file, weights_path} <- find_weights_file(repository, path) do
      tensors = load_tensors(weights_file, weights_path, opts)

      embedding_weight =
        (tensors["embedding.weight"] || tensors["weight"])
        |> Nx.to_tensor()
        |> cast_param(opts[:type])
        |> allocate_param(opts[:backend])

      {vocab_size, embedding_dim} = Nx.shape(embedding_weight)
      layer_name = "#{path}.embedding"

      model =
        "input_ids"
        |> Axon.input(optional: false)
        |> Axon.embedding(vocab_size, embedding_dim, name: layer_name)
        |> Axon.nx(fn hidden_state ->
          %{hidden_state: hidden_state}
        end)

      params = %Axon.ModelState{
        data: %{layer_name => %{"kernel" => embedding_weight}},
        parameters: %{layer_name => ["kernel"]},
        frozen_parameters: %{},
        state: %{}
      }

      {:ok, %{model: model, params: params, spec: nil}}
    end
  end

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
