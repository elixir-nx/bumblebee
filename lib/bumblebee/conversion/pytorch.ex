defmodule Bumblebee.Conversion.PyTorch do
  @moduledoc false

  require Logger

  alias Bumblebee.Utils

  @doc """
  Loads parameters from a PyTorch model state dictionary.

  This function expects files created with `torch.save(model.state_dict(), path)`,
  as described in [the documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

  ## Options

    * `:base_model_prefix` - the base model name in layer names.
      Allows for loading base model parameters into specialized model
      and vice versa

  """
  @spec load_params!(Axon.t(), Path.t(), keyword()) :: map()
  def load_params!(model, path, opts \\ []) do
    prefix = opts[:base_model_prefix]

    pytorch_state = Bumblebee.Conversion.PyTorch.Loader.load!(path)

    unless state_dict?(pytorch_state) do
      raise "expected a serialized model state dictionary at #{path}, but got: #{inspect(pytorch_state)}"
    end

    {params, diff} = init_params(model, pytorch_state, prefix)

    log_model_diff(diff)

    params
  end

  defp state_dict?(%{} = dict) when not is_struct(dict) do
    Enum.all?(dict, fn {key, value} -> is_binary(key) and is_struct(value, Nx.Tensor) end)
  end

  defp state_dict?(_other), do: false

  defp init_params(model, pytorch_state, prefix) do
    layers =
      model
      |> Utils.Axon.nodes_with_names()
      |> Enum.filter(fn {layer, _name} -> layer.parameters != [] end)

    prefixed = check_prefix(prefix, layers, pytorch_state)

    diff = %{missing: [], mismatched: [], used_keys: []}

    {params, diff} =
      Enum.map_reduce(layers, diff, fn {layer, layer_name}, diff ->
        source_layer_name = source_layer_name(layer_name, prefix, prefixed)

        {params, diff} =
          Enum.reduce(layer.parameters, {[], diff}, fn param, {params, diff} ->
            {value, diff} =
              case param_from_pytorch(layer.op_name, param.name, pytorch_state, source_layer_name) do
                {:ok, value, keys} ->
                  diff = prepend(diff, :used_keys, keys)

                  case verify_param_shape(param, value) do
                    :ok ->
                      {value, diff}

                    {:error, expected, actual} ->
                      {nil, prepend(diff, :mismatched, [{layer_name, param, expected, actual}])}
                  end

                :error ->
                  {nil, prepend(diff, :missing, [{layer_name, param}])}
              end

            value = value || Utils.Axon.init_param(layer, param)
            {[{param.name, value} | params], diff}
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

  defp check_prefix(nil, _layers, _pytorch_state), do: %{target: false, source: false}

  defp check_prefix(prefix, layers, pytorch_state) do
    target_prefixed? =
      Enum.any?(layers, fn {_, name} ->
        String.starts_with?(name, prefix <> ".")
      end)

    source_prefixed? =
      Enum.any?(pytorch_state, fn {key, _} ->
        String.starts_with?(key, prefix <> ".")
      end)

    %{target: target_prefixed?, source: source_prefixed?}
  end

  defp source_layer_name(target_layer_name, prefix, prefixed) do
    case prefixed do
      %{target: false, source: true} ->
        prefix <> "." <> target_layer_name

      %{target: true, source: false} ->
        String.replace_prefix(target_layer_name, prefix <> ".", "")

      _ ->
        target_layer_name
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
      [in_channels, out_channels | kernel_spacials] = Nx.axes(kernel)
      kernel = Nx.transpose(kernel, axes: [out_channels, in_channels | kernel_spacials])
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

  defp verify_param_shape(param, value) do
    case param.shape do
      {:tuple, shapes} ->
        {List.to_tuple(shapes),
         value
         |> Tuple.to_list()
         |> Enum.map(&Nx.shape/1)
         |> List.to_tuple()}

      shape ->
        {shape, Nx.shape(value)}
    end
    |> case do
      {shape, shape} -> :ok
      {expected, actual} -> {:error, expected, actual}
    end
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
