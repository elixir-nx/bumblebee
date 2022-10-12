defmodule Bumblebee.Conversion.PyTorch do
  @moduledoc false

  require Logger

  alias Bumblebee.Utils

  @doc """
  Loads parameters from a PyTorch model state dictionary.

  This function expects files created with `torch.save(model.state_dict(), path)`,
  as described in [the documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).
  """
  @spec load_params!(Axon.t(), map(), Path.t()) :: map()
  def load_params!(model, input_template, path) do
    pytorch_state = Bumblebee.Conversion.PyTorch.Loader.load!(path)

    unless state_dict?(pytorch_state) do
      raise "expected a serialized model state dictionary at #{path}, but got: #{inspect(pytorch_state)}"
    end

    params_expr = Axon.trace_init(model, input_template)

    {params, diff} = init_params(model, params_expr, pytorch_state)

    params =
      if diff.missing == [] do
        params
      else
        {init_fun, _} = Axon.build(model)
        init_fun.(input_template, params)
      end

    log_model_diff(diff)

    params
  end

  defp state_dict?(%{} = dict) when not is_struct(dict) do
    Enum.all?(dict, fn {key, value} -> is_binary(key) and is_struct(value, Nx.Tensor) end)
  end

  defp state_dict?(_other), do: false

  defp init_params(model, params_expr, pytorch_state) do
    layers =
      model
      |> Utils.Axon.nodes_with_names()
      |> Enum.filter(fn {layer, _name} -> layer.parameters != [] end)

    prefixes = infer_prefixes(layers, pytorch_state)

    diff = %{missing: [], mismatched: [], used_keys: []}

    {params, diff} =
      layers
      |> Enum.filter(fn {_layer, layer_name} -> params_expr[layer_name] end)
      |> Enum.map_reduce(diff, fn {layer, layer_name}, diff ->
        source_layer_name = source_layer_name(layer_name, prefixes)

        {params, diff} =
          Enum.reduce(layer.parameters, {[], diff}, fn param, {params, diff} ->
            param_expr = params_expr[layer_name][param.name]

            {value, diff} =
              case param_from_pytorch(layer.op_name, param.name, pytorch_state, source_layer_name) do
                {:ok, value, keys} ->
                  diff = prepend(diff, :used_keys, keys)

                  case verify_param_shape(param_expr, value) do
                    :ok ->
                      {value, diff}

                    {:error, expected, actual} ->
                      {nil, prepend(diff, :mismatched, [{layer_name, param, expected, actual}])}
                  end

                :error ->
                  {nil, prepend(diff, :missing, [{layer_name, param}])}
              end

            if value do
              {[{param.name, value} | params], diff}
            else
              {params, diff}
            end
          end)

        ignored_keys = ignored_params(pytorch_state, source_layer_name)
        diff = prepend(diff, :used_keys, ignored_keys)
        {{layer_name, Map.new(params)}, diff}
      end)

    params = Map.new(params)

    diff = %{
      missing: Enum.reverse(diff.missing),
      mismatched: Enum.reverse(diff.mismatched),
      unused_keys: Enum.sort(Map.keys(pytorch_state) -- diff.used_keys)
    }

    {params, diff}
  end

  defp prepend(diff, key, values), do: Map.update!(diff, key, &(values ++ &1))

  defp infer_prefixes(layers, pytorch_state) do
    target_names = for {_, name} <- layers, do: name

    source_names =
      for {key, _} <- pytorch_state, uniq: true do
        # Drop parameter name
        key |> String.split(".") |> Enum.drop(-1) |> Enum.join(".")
      end

    source_prefix = maybe_prefix(target_names, source_names)
    target_prefix = maybe_prefix(source_names, target_names)

    %{target: target_prefix, source: source_prefix}
  end

  # If a subset of `names` is present in `prefixed_names` under the
  # same prefix, finds that prefix.
  defp maybe_prefix(names, prefixed_names) do
    names
    |> Enum.map(fn name ->
      suffix = "." <> name

      for prefixed_name <- prefixed_names,
          String.ends_with?(prefixed_name, suffix),
          do: String.replace_suffix(prefixed_name, suffix, "")
    end)
    |> Enum.reject(&(&1 == []))
    |> case do
      [] ->
        nil

      prefixes ->
        prefixes
        |> Enum.map(&MapSet.new/1)
        |> Enum.reduce(&MapSet.intersection/2)
        |> Enum.to_list()
        |> case do
          [prefix] -> prefix
        end
    end
  end

  defp source_layer_name(target_layer_name, prefixes) do
    case prefixes do
      %{target: prefix, source: prefix} ->
        target_layer_name

      %{target: nil, source: source_prefix} ->
        source_prefix <> "." <> target_layer_name

      %{target: target_prefix, source: nil} ->
        String.replace_prefix(target_layer_name, target_prefix <> ".", "")
    end
  end

  defp log_model_diff(%{missing: missing, mismatched: mismatched, unused_keys: unused_keys}) do
    if missing != [] do
      missing_keys =
        Enum.map(missing, fn {layer_name, param} -> layer_name <> "." <> param.name end)

      Logger.debug("the following parameters were missing:\n\n#{format_list(missing_keys)}\n")
    end

    if unused_keys != [] do
      Logger.debug(
        "the following PyTorch parameters were unused:\n\n#{format_list(unused_keys)}\n"
      )
    end

    if mismatched != [] do
      mismatched_keys =
        Enum.map(mismatched, fn {layer_name, param, expected_shape, actual_shape} ->
          "#{layer_name}.#{param.name} (expected #{inspect(expected_shape)}, got: #{inspect(actual_shape)})"
        end)

      Logger.debug(
        "the following parameters were ignored, because of non-matching shape:\n\n#{format_list(mismatched_keys)}\n"
      )
    end
  end

  defp format_list(items), do: Enum.map_join(items, "\n", &("  * " <> &1))

  defp param_from_pytorch(:dense, "kernel", pytorch_state, layer_name) do
    with {:ok, kernel, key} <- lookup_param(pytorch_state, layer_name, ["weight"]) do
      [out_features, in_features] = Nx.axes(kernel)
      kernel = Nx.transpose(kernel, axes: [in_features, out_features])
      {:ok, kernel, [key]}
    end
  end

  defp param_from_pytorch(:conv_transpose, "kernel", pytorch_state, layer_name) do
    with {:ok, kernel, key} <- lookup_param(pytorch_state, layer_name, ["weight"]) do
      [in_channels, out_channels | kernel_spatials] = Nx.axes(kernel)
      kernel = Nx.transpose(kernel, axes: [out_channels, in_channels | kernel_spatials])
      {:ok, kernel, [key]}
    end
  end

  defp param_from_pytorch(:lstm, "bias", pytorch_state, layer_name) do
    with {:ok, bias_hh, key1} <- lookup_param(pytorch_state, layer_name, ["bias_hh"]),
         {:ok, bias_ih, key2} <- lookup_param(pytorch_state, layer_name, ["bias_ih"]) do
      bias = Nx.add(bias_ih, bias_hh)
      bias = Nx.reshape(bias, {4, :auto})
      {:ok, {bias[0], bias[1], bias[2], bias[3]}, [key1, key2]}
    end
  end

  defp param_from_pytorch(:lstm, "input_kernel", pytorch_state, layer_name) do
    with {:ok, weight_ih, key} <- lookup_param(pytorch_state, layer_name, ["weight_ih"]) do
      weight_ih = weight_ih |> unflatten_leading(4) |> Nx.transpose(axes: [0, 2, 1])
      {:ok, {weight_ih[0], weight_ih[1], weight_ih[2], weight_ih[3]}, [key]}
    end
  end

  defp param_from_pytorch(:lstm, "hidden_kernel", pytorch_state, layer_name) do
    with {:ok, weight_hh, key} <- lookup_param(pytorch_state, layer_name, ["weight_hh"]) do
      weight_hh = weight_hh |> unflatten_leading(4) |> Nx.transpose(axes: [0, 2, 1])
      {:ok, {weight_hh[0], weight_hh[1], weight_hh[2], weight_hh[3]}, [key]}
    end
  end

  defp param_from_pytorch(:gru, "bias", pytorch_state, layer_name) do
    with {:ok, bias_hh, key1} <- lookup_param(pytorch_state, layer_name, ["bias_hh"]),
         {:ok, bias_ih, key2} <- lookup_param(pytorch_state, layer_name, ["bias_ih"]) do
      bias_hh = unflatten_leading(bias_hh, 3)
      bias_ih = unflatten_leading(bias_ih, 3)

      bias =
        {Nx.add(bias_ih[0], bias_hh[0]), Nx.add(bias_ih[1], bias_hh[1]), bias_ih[2], bias_hh[2]}

      {:ok, bias, [key1, key2]}
    end
  end

  defp param_from_pytorch(:gru, "input_kernel", pytorch_state, layer_name) do
    with {:ok, weight_ih, key} <- lookup_param(pytorch_state, layer_name, ["weight_ih"]) do
      weight_ih = weight_ih |> unflatten_leading(3) |> Nx.transpose(axes: [0, 2, 1])
      {:ok, {weight_ih[0], weight_ih[1], weight_ih[2]}, [key]}
    end
  end

  defp param_from_pytorch(:gru, "hidden_kernel", pytorch_state, layer_name) do
    with {:ok, weight_hh, key} <- lookup_param(pytorch_state, layer_name, ["weight_hh"]) do
      weight_hh = weight_hh |> unflatten_leading(3) |> Nx.transpose(axes: [0, 2, 1])
      {:ok, {weight_hh[0], weight_hh[1], weight_hh[2]}, [key]}
    end
  end

  defp param_from_pytorch(_op, param_name, pytorch_state, layer_name) do
    pytorch_names =
      case param_name do
        # PyTorch uses "weight" instead of "kernel" everywhere
        "kernel" -> ["weight"]
        # For normalization layers PyTorch uses "weight" and "bias",
        # however we check the other ones just in case.
        #
        # [1]: https://github.com/huggingface/transformers/pull/11394
        # [2]: https://github.com/naver/sqlova/issues/1#issuecomment-463481275
        "gamma" -> ["weight", "gamma"]
        "beta" -> ["bias", "beta"]
        # Running averages in normalization layers
        "mean" -> ["running_mean"]
        "var" -> ["running_var"]
        name -> [name]
      end

    with {:ok, value, key} <- lookup_param(pytorch_state, layer_name, pytorch_names),
         do: {:ok, value, [key]}
  end

  defp lookup_param(pytorch_state, layer_name, pytorch_names) do
    Enum.find_value(pytorch_names, :error, fn pytorch_name ->
      pytorch_key = layer_name <> "." <> pytorch_name

      if value = pytorch_state[pytorch_key] do
        {:ok, value, pytorch_key}
      end
    end)
  end

  defp verify_param_shape(param_expr, value) do
    case {expr_shape(param_expr), expr_shape(value)} do
      {shape, shape} -> :ok
      {expected, actual} -> {:error, expected, actual}
    end
  end

  defp expr_shape(expr) do
    Utils.Nx.map(expr, &Nx.shape/1)
  end

  defp ignored_params(pytorch_state, layer_name) do
    ignored = ["num_batches_tracked"]

    for name <- ignored,
        key = layer_name <> "." <> name,
        Map.has_key?(pytorch_state, key),
        do: key
  end

  defp unflatten_leading(tensor, axis_size) do
    shape =
      tensor
      |> Nx.shape()
      |> Tuple.insert_at(0, axis_size)
      |> put_elem(1, :auto)

    Nx.reshape(tensor, shape)
  end
end
