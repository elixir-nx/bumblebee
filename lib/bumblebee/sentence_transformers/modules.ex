defmodule Bumblebee.SentenceTransformers.Modules do
  @moduledoc false

  alias Axon.MixedPrecision.Policy
  alias Bumblebee.SentenceTransformers.Downloader
  alias Bumblebee.SentenceTransformers.Pipeline

  @weights_filenames ["model.safetensors", "pytorch_model.bin"]

  @doc """
  Builds a single pipeline module step on top of the current computation node.
  """
  def build(repository, module, node, attention_mask, params_data, params_parameters, opts) do
    case Pipeline.normalize_module_type(module["type"]) do
      "Pooling" ->
        build_pooling(repository, module, node, attention_mask, params_data, params_parameters)

      "Dense" ->
        build_dense(repository, module, node, params_data, params_parameters, opts)

      "FusedDense" ->
        build_fused_dense(repository, module, node, params_data, params_parameters, opts)

      "Normalize" ->
        build_normalize(module, node, params_data, params_parameters)

      "Dropout" ->
        build_dropout(node, params_data, params_parameters)

      "LayerNorm" ->
        build_layer_norm(repository, module, node, params_data, params_parameters, opts)

      "WeightedLayerPooling" ->
        build_weighted_layer_pooling(
          repository,
          module,
          node,
          params_data,
          params_parameters,
          opts
        )

      other ->
        {:error, "unsupported SentenceTransformers module #{inspect(other)}"}
    end
  end

  @doc """
  Builds a Pooling module on top of the given node.
  """
  def build_pooling(
        repository,
        module,
        node,
        attention_mask,
        params_data,
        params_parameters
      ) do
    with {:ok, config} <- load_pooling_config(repository, module),
         {:ok, node} <- pooling_layer(node, attention_mask, config, module["path"]) do
      requires_prompt_length = Map.get(config, "include_prompt", true) == false
      {:ok, node, params_data, params_parameters, requires_prompt_length}
    end
  end

  defp load_pooling_config(_repository, %{"config" => config}) when is_map(config) do
    {:ok, config}
  end

  defp load_pooling_config(_repository, %{"path" => "default_pooling"} = module) do
    mode = module["pooling_mode"] || "mean"
    {:ok, %{"pooling_mode" => mode}}
  end

  defp load_pooling_config(repository, %{"path" => path}) do
    load_module_config(repository, path)
  end

  @doc """
  Builds a Dense module on top of the given node.
  """
  def build_dense(
        repository,
        %{"path" => path},
        node,
        params_data,
        params_parameters,
        opts
      ) do
    with {:ok, config} <- load_module_config(repository, path),
         {:ok, activation} <- parse_activation(config["activation_function"], path),
         {:ok, weights_file, weights_path} <- find_weights_file(repository, path) do
      in_features = config["in_features"]
      out_features = config["out_features"]
      bias? = Map.get(config, "bias", true)
      use_residual? = Map.get(config, "use_residual", false)

      out_node =
        node
        |> Axon.dense(out_features, use_bias: bias?, name: path)
        |> maybe_activation(activation)

      tensors = load_tensors(weights_file, weights_path, opts)
      layer_params = load_dense_params(tensors, bias?, opts)

      {params_data, params_parameters} =
        put_layer_params(params_data, params_parameters, path, layer_params)

      {node, params_data, params_parameters} =
        maybe_dense_residual(
          out_node,
          node,
          use_residual?,
          in_features,
          out_features,
          tensors,
          path,
          params_data,
          params_parameters,
          opts
        )

      {:ok, node, params_data, params_parameters, false}
    end
  end

  defp load_dense_params(tensors, bias?, opts) do
    kernel =
      tensors["linear.weight"]
      |> Nx.to_tensor()
      |> Nx.transpose()
      |> cast_param(opts[:type])
      |> allocate_param(opts[:backend])

    if bias? do
      bias_tensor =
        tensors["linear.bias"]
        |> Nx.to_tensor()
        |> cast_param(opts[:type])
        |> allocate_param(opts[:backend])

      %{"kernel" => kernel, "bias" => bias_tensor}
    else
      %{"kernel" => kernel}
    end
  end

  defp maybe_dense_residual(
         out_node,
         _node,
         false,
         _in_features,
         _out_features,
         _tensors,
         _path,
         params_data,
         params_parameters,
         _opts
       ) do
    {out_node, params_data, params_parameters}
  end

  defp maybe_dense_residual(
         out_node,
         node,
         true,
         features,
         features,
         _tensors,
         path,
         params_data,
         params_parameters,
         _opts
       ) do
    {Axon.add(out_node, node, name: "#{path}.add_residual"), params_data, params_parameters}
  end

  defp maybe_dense_residual(
         out_node,
         node,
         true,
         _in_features,
         out_features,
         tensors,
         path,
         params_data,
         params_parameters,
         opts
       ) do
    res_name = "#{path}.residual"
    res_node = Axon.dense(node, out_features, use_bias: false, name: res_name)

    res_kernel =
      tensors["residual.weight"]
      |> Nx.to_tensor()
      |> Nx.transpose()
      |> cast_param(opts[:type])
      |> allocate_param(opts[:backend])

    {params_data, params_parameters} =
      put_layer_params(params_data, params_parameters, res_name, %{"kernel" => res_kernel})

    {Axon.add(out_node, res_node, name: "#{path}.add_residual"), params_data, params_parameters}
  end

  @doc """
  Builds a FusedDense module on top of the given node.
  """
  def build_fused_dense(
        repository,
        module,
        node,
        params_data,
        params_parameters,
        opts
      ) do
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

      {params_data, params_parameters} =
        put_layer_params(params_data, params_parameters, layer_name, %{"kernel" => k_fused})

      {:ok, node, params_data, params_parameters, false}
    end
  end

  @doc """
  Builds a Normalize module on top of the given node.
  """
  def build_normalize(%{"path" => path}, node, params_data, params_parameters) do
    node = Axon.nx(node, &Bumblebee.Utils.Nx.normalize/1, name: "#{path}.normalize")
    {:ok, node, params_data, params_parameters, false}
  end

  @doc """
  Builds a Dropout module (no-op at inference).
  """
  def build_dropout(node, params_data, params_parameters) do
    {:ok, node, params_data, params_parameters, false}
  end

  @doc """
  Builds a LayerNorm module on top of the given node.
  """
  def build_layer_norm(
        repository,
        %{"path" => path},
        node,
        params_data,
        params_parameters,
        opts
      ) do
    with {:ok, config} <- load_module_config(repository, path),
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

      {params_data, params_parameters} =
        put_layer_params(params_data, params_parameters, layer_name, %{
          "gamma" => gamma,
          "beta" => beta
        })

      {:ok, node, params_data, params_parameters, false}
    end
  end

  @doc """
  Builds a WeightedLayerPooling module on top of the given node.
  """
  def build_weighted_layer_pooling(
        repository,
        %{"path" => path},
        node,
        params_data,
        params_parameters,
        opts
      ) do
    with {:ok, config} <- load_module_config(repository, path) do
      layer_start = config["layer_start"] || 4
      num_hidden_layers = config["num_hidden_layers"] || 12
      num_weights = max(num_hidden_layers + 1 - layer_start, 1)
      layer_name = path

      weights = load_layer_weights(repository, path, num_weights, opts)

      {params_data, params_parameters} =
        put_layer_params(params_data, params_parameters, layer_name, %{"layer_weights" => weights})

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
  end

  defp load_layer_weights(repository, path, num_weights, opts) do
    raw_weights =
      case find_weights_file(repository, path) do
        {:ok, weights_file, weights_path} ->
          tensors = load_tensors(weights_file, weights_path, opts)
          tensors["layer_weights"] || tensors["weight"] || Nx.broadcast(1.0, {num_weights})

        _ ->
          Nx.broadcast(1.0, {num_weights})
      end

    raw_weights
    |> Nx.to_tensor()
    |> cast_param(opts[:type])
    |> allocate_param(opts[:backend])
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

  @doc """
  Constructs a default mean-pooling layer.
  """
  def default_pooling(node, attention_mask, name \\ "default_pooling") do
    config = %{"pooling_mode_mean_tokens" => true}
    pooling_layer(node, attention_mask, config, name)
  end

  @doc """
  Constructs a pooling layer from a configuration map.
  """
  def pooling_layer(hidden_state, attention_mask, config, name) do
    include_prompt = Map.get(config, "include_prompt", true)

    case extract_pooling_modes(config) do
      [] ->
        {:error, "no supported pooling mode found in #{name}"}

      modes ->
        layer = build_pooling_layer(hidden_state, attention_mask, modes, include_prompt, name)
        {:ok, layer}
    end
  end

  defp extract_pooling_modes(config) do
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

  defp maybe_activation(node, nil), do: node
  defp maybe_activation(node, activation), do: Axon.activation(node, activation)

  @doc false
  def load_module_config(repository, path) do
    with {:ok, config_path} <-
           Downloader.download_file(repository, Path.join(path, "config.json")) do
      decode_json(config_path)
    end
  end

  def find_weights_file(repository, dir) do
    Enum.find_value(@weights_filenames, fn filename ->
      path = Path.join(dir, filename)

      case Downloader.download_file(repository, path) do
        {:ok, downloaded_path} -> {:ok, path, downloaded_path}
        _ -> nil
      end
    end) || {:error, "could not find parameters file in #{dir}"}
  end

  def load_tensors(weights_file, weights_path, opts) do
    case Path.extname(weights_file) do
      ".safetensors" ->
        reader = opts[:safetensors_reader] || (&Safetensors.read!(&1, lazy: true))
        reader.(weights_path)

      _ ->
        Bumblebee.Conversion.PyTorchLoader.load!(weights_path)
    end
  end

  def cast_param(tensor, nil), do: tensor

  def cast_param(tensor, %Policy{params: type}) do
    Nx.as_type(tensor, type)
  end

  def cast_param(tensor, type) do
    type = Nx.Type.normalize!(type)
    Nx.as_type(tensor, type)
  end

  def allocate_param(tensor, nil), do: tensor

  def allocate_param(tensor, backend) do
    Nx.with_default_backend(backend, fn -> Nx.backend_copy(tensor) end)
  end

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

  defp put_layer_params(params_data, params_parameters, layer_name, layer_params) do
    params_data = Map.put(params_data, layer_name, layer_params)

    params_parameters =
      if params_parameters do
        Map.put(params_parameters, layer_name, Map.keys(layer_params))
      end

    {params_data, params_parameters}
  end
end
