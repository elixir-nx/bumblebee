defmodule Bumblebee.Conversion.PyTorch do
  @moduledoc false

  require Logger

  alias Bumblebee.Utils

  @doc """
  Loads parameters from a PyTorch model state dictionary.

  This function expects files created with `torch.save(model.state_dict(), path)`,
  as described in [the documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

  ## Options

    * `:log_params_diff` - whether to log missing, mismatched and unused
      parameters. By default diff is logged only if some parameters
      cannot be loaded

    * `:backend` - the backend to allocate the tensors on. It is either
      an atom or a tuple in the shape `{backend, options}`

    * `:params_mapping` - a map describing layers and params relationship
      between the Axon model and the PyTorch state. For more details see
      `Bumblebee.HuggingFace.Transformers.Model.params_mapping/1`

    * `:loader_fun` - a 1-arity function that takes a path argument
      and loads the params file. Defaults to
      `Bumblebee.Conversion.PyTorch.Loader.load!/1`

  """
  @spec load_params!(Axon.t(), map(), Path.t() | list(Path.t()), keyword()) :: map()
  def load_params!(model, input_template, path, opts \\ []) do
    opts =
      opts
      |> Keyword.validate!([
        :log_params_diff,
        :backend,
        params_mapping: %{},
        loader_fun: &Bumblebee.Conversion.PyTorch.Loader.load!/1
      ])

    with_default_backend(opts[:backend], fn ->
      pytorch_state =
        path
        |> List.wrap()
        |> Enum.map(fn path ->
          pytorch_state = opts[:loader_fun].(path)

          unless state_dict?(pytorch_state) do
            raise "expected a serialized model state dictionary at #{path}, but got: #{inspect(pytorch_state)}"
          end

          pytorch_state
        end)
        |> Enum.reduce(&Map.merge/2)

      params_expr = Axon.trace_init(model, input_template)

      {params, diff} = init_params(model, params_expr, pytorch_state, opts[:params_mapping])

      params_complete? = diff.missing == [] and diff.mismatched == []

      params =
        if params_complete? do
          params
        else
          {init_fun, _} = Axon.build(model, compiler: Nx.Defn.Evaluator)
          init_fun.(input_template, params)
        end

      if Keyword.get(opts, :log_params_diff, not params_complete?) do
        log_params_diff(diff)
      end

      params
    end)
  end

  defp with_default_backend(nil, fun), do: fun.()
  defp with_default_backend(backend, fun), do: Nx.with_default_backend(backend, fun)

  defp state_dict?(%{} = dict) when not is_struct(dict) do
    Enum.all?(dict, fn {key, value} ->
      is_binary(key) and Nx.LazyContainer.impl_for(value) != nil
    end)
  end

  defp state_dict?(_other), do: false

  defp init_params(model, params_expr, pytorch_state, params_mapping) do
    layers =
      model
      |> Utils.Axon.nodes_with_names()
      |> Enum.filter(fn {layer, _name} -> layer.parameters != [] end)

    prefixes = infer_prefixes(layers, pytorch_state, params_mapping)

    diff = %{missing: [], mismatched: [], used_keys: []}

    {params, diff} =
      layers
      |> Enum.filter(fn {_layer, layer_name} -> params_expr[layer_name] end)
      |> Enum.map_reduce(diff, fn {layer, layer_name}, diff ->
        params_source = params_source(layer_name, prefixes, params_mapping)

        {params, diff} =
          Enum.reduce(layer.parameters, {[], diff}, fn param, {params, diff} ->
            param_expr = params_expr[layer_name][param.name]

            {sources, builder_fun} =
              case params_source do
                %{} = param_builders ->
                  if param_builder = param_builders[param.name] do
                    param_builder
                  else
                    raise "no matching mapping found for parameter #{inspect(param.name)} in #{inspect(param_builders)}"
                  end

                source_layer_name
                when is_binary(source_layer_name) or
                       is_list(source_layer_name) ->
                  default_layer_param_builder(layer, param.name, source_layer_name)
              end

            {all_sources_found?, source_values, source_keys} =
              for source <- sources, reduce: {true, [], []} do
                {all_found?, values, keys} ->
                  # Source can be either {layer_name, param_name}, or
                  # a list of these, to find any match
                  source
                  |> List.wrap()
                  |> Enum.find_value(fn {source_layer_name, source_param_name} ->
                    lookup_param(pytorch_state, source_layer_name, source_param_name)
                  end)
                  |> case do
                    {value, key} -> {all_found?, [value | values], [key | keys]}
                    nil -> {false, values, keys}
                  end
              end

            diff = prepend(diff, :used_keys, source_keys)

            {value, diff} =
              if all_sources_found? do
                source_values = Enum.map(source_values, &Nx.to_tensor/1)
                value = builder_fun.(Enum.reverse(source_values))

                case verify_param_shape(param_expr, value) do
                  :ok ->
                    value = ensure_type(param_expr, value)
                    {value, diff}

                  {:error, expected, actual} ->
                    {nil, prepend(diff, :mismatched, [{layer_name, param, expected, actual}])}
                end
              else
                {nil, prepend(diff, :missing, [{layer_name, param}])}
              end

            if value do
              {[{param.name, value} | params], diff}
            else
              {params, diff}
            end
          end)

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

  defp infer_prefixes(layers, pytorch_state, params_mapping) do
    # Note: target refers to the parameters we are initializing, while
    # source refers to the state we are loading from

    target_names = for {_, name} <- layers, do: name

    source_names =
      for {key, _} <- pytorch_state, uniq: true do
        # Drop parameter name
        key |> String.split(".") |> Enum.drop(-1) |> Enum.join(".")
      end

    target_templates = Map.keys(params_mapping)

    source_templates =
      Enum.flat_map(params_mapping, fn
        {_target_template, %{} = param_builders} ->
          for {_target_param_name, {sources, _builder_fun}} <- param_builders,
              ref_or_refs <- sources,
              {source_template, _source_param_name} <- List.wrap(ref_or_refs),
              do: source_template

        {_target_template, source_templates} when is_list(source_templates) ->
          source_templates

        {_target_template, source_template} when is_binary(source_template) ->
          [source_template]
      end)

    target_template_prefix = maybe_prefix(target_names, target_templates)
    target_name_prefix = maybe_prefix(target_templates, target_names)
    source_template_prefix = maybe_prefix(source_names, source_templates)
    source_name_prefix = maybe_prefix(source_templates, source_names)

    %{
      target_template: target_template_prefix,
      target_name: target_name_prefix,
      source_template: source_template_prefix,
      source_name: source_name_prefix
    }
  end

  # If a subset of `names` is present in `prefixed_names` under the
  # same prefix, finds that prefix.
  defp maybe_prefix(names, prefixed_names) do
    (names -- prefixed_names)
    |> Enum.map(fn name ->
      suffix = "." <> name

      for prefixed_name <- prefixed_names,
          String.ends_with?(prefixed_name, suffix),
          do: String.replace_suffix(prefixed_name, suffix, ""),
          into: MapSet.new()
    end)
    |> Enum.reject(&Enum.empty?/1)
    |> case do
      [] ->
        nil

      prefix_sets ->
        prefix_sets
        |> Enum.reduce(&MapSet.intersection/2)
        |> Enum.to_list()
        |> case do
          [prefix] -> prefix
        end
    end
  end

  defp params_source(target_layer_name, prefixes, params_mapping) do
    layer_name = change_prefix(target_layer_name, prefixes.target_name, prefixes.target_template)

    params_source =
      case params_mapping do
        %{^layer_name => params_source} ->
          params_source

        mapping ->
          params_source =
            Enum.find_value(mapping, fn {target_template, params_source} ->
              if substitutes = match_template(layer_name, target_template) do
                Bumblebee.HuggingFace.Transformers.Utils.map_params_source_layer_names(
                  params_source,
                  &fill_template(&1, substitutes)
                )
              end
            end)

          unless params_source do
            raise "no matching mapping found for layer #{inspect(layer_name)} in #{inspect(mapping)}"
          end

          params_source
      end

    Bumblebee.HuggingFace.Transformers.Utils.map_params_source_layer_names(
      params_source,
      &change_prefix(&1, prefixes.source_template, prefixes.source_name)
    )
  end

  defp change_prefix(layer_name, current_prefix, new_prefix) do
    layer_name =
      if current_prefix do
        String.replace_prefix(layer_name, current_prefix <> ".", "")
      else
        layer_name
      end

    if new_prefix do
      new_prefix <> "." <> layer_name
    else
      layer_name
    end
  end

  defp match_template(name, template), do: match_template(name, template, %{})

  defp match_template(<<_, _::binary>> = name, <<"{", template::binary>>, substitutes) do
    [value, name] = String.split(name, ".", parts: 2)
    [key, template] = String.split(template, "}.", parts: 2)
    match_template(name, template, put_in(substitutes[key], value))
  end

  defp match_template(<<h, name::binary>>, <<h, template::binary>>, substitutes) do
    match_template(name, template, substitutes)
  end

  defp match_template(<<>>, <<>>, substitutes), do: substitutes
  defp match_template(_name, _template, _substitutes), do: nil

  defp fill_template(template, substitutes), do: fill_template(template, substitutes, <<>>)

  defp fill_template(<<>>, _substitutes, name), do: name

  defp fill_template(<<"{", template::binary>>, substitutes, name) do
    [key, template] = String.split(template, "}", parts: 2)
    value = Map.fetch!(substitutes, key)
    fill_template(template, substitutes, <<name::binary, value::binary>>)
  end

  defp fill_template(<<h, template::binary>>, substitutes, name) do
    fill_template(template, substitutes, <<name::binary, h>>)
  end

  defp log_params_diff(%{missing: missing, mismatched: mismatched, unused_keys: unused_keys}) do
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

  defp default_layer_param_builder(%{op_name: :dense}, "kernel", layer_name) do
    {[param_refs(layer_name, "weight")],
     fn [kernel] ->
       [out_features, in_features] = Nx.axes(kernel)
       Nx.transpose(kernel, axes: [in_features, out_features])
     end}
  end

  defp default_layer_param_builder(layer, "kernel", layer_name)
       when layer.op_name in [:conv, :depthwise_conv] do
    {[param_refs(layer_name, "weight")],
     fn [kernel] ->
       [out_channels, in_channels | kernel_spatials] = Nx.axes(kernel)

       case layer.opts[:channels] do
         :first -> kernel
         :last -> Nx.transpose(kernel, axes: kernel_spatials ++ [in_channels, out_channels])
       end
     end}
  end

  defp default_layer_param_builder(%{op_name: :conv_transpose} = layer, "kernel", layer_name) do
    {[param_refs(layer_name, "weight")],
     fn [kernel] ->
       [in_channels, out_channels | kernel_spatials] = Nx.axes(kernel)

       case layer.opts[:channels] do
         :first -> Nx.transpose(kernel, axes: [out_channels, in_channels | kernel_spatials])
         :last -> Nx.transpose(kernel, axes: kernel_spatials ++ [in_channels, out_channels])
       end
     end}
  end

  defp default_layer_param_builder(%{op_name: :lstm}, "bias", layer_name) do
    {[param_refs(layer_name, "bias_hh"), param_refs(layer_name, "bias_ih")],
     fn [bias_hh, bias_ih] ->
       bias = Nx.add(bias_ih, bias_hh)
       bias = Nx.reshape(bias, {4, :auto})
       {bias[0], bias[1], bias[2], bias[3]}
     end}
  end

  defp default_layer_param_builder(%{op_name: :lstm}, "input_kernel", layer_name) do
    {[param_refs(layer_name, "weight_ih")],
     fn [weight_ih] ->
       weight_ih = weight_ih |> unflatten_leading(4) |> Nx.transpose(axes: [0, 2, 1])
       {weight_ih[0], weight_ih[1], weight_ih[2], weight_ih[3]}
     end}
  end

  defp default_layer_param_builder(%{op_name: :lstm}, "hidden_kernel", layer_name) do
    {[param_refs(layer_name, "weight_hh")],
     fn [weight_hh] ->
       weight_hh = weight_hh |> unflatten_leading(4) |> Nx.transpose(axes: [0, 2, 1])
       {weight_hh[0], weight_hh[1], weight_hh[2], weight_hh[3]}
     end}
  end

  defp default_layer_param_builder(%{op_name: :gru}, "bias", layer_name) do
    {[param_refs(layer_name, "bias_hh"), param_refs(layer_name, "bias_ih")],
     fn [bias_hh, bias_ih] ->
       bias_hh = unflatten_leading(bias_hh, 3)
       bias_ih = unflatten_leading(bias_ih, 3)
       {Nx.add(bias_ih[0], bias_hh[0]), Nx.add(bias_ih[1], bias_hh[1]), bias_ih[2], bias_hh[2]}
     end}
  end

  defp default_layer_param_builder(%{op_name: :gru}, "input_kernel", layer_name) do
    {[param_refs(layer_name, "weight_ih")],
     fn [weight_ih] ->
       weight_ih = weight_ih |> unflatten_leading(4) |> Nx.transpose(axes: [0, 2, 1])
       {weight_ih[0], weight_ih[1], weight_ih[2]}
     end}
  end

  defp default_layer_param_builder(%{op_name: :gru}, "hidden_kernel", layer_name) do
    {[param_refs(layer_name, "weight_hh")],
     fn [weight_hh] ->
       weight_hh = weight_hh |> unflatten_leading(3) |> Nx.transpose(axes: [0, 2, 1])
       {weight_hh[0], weight_hh[1], weight_hh[2]}
     end}
  end

  defp default_layer_param_builder(_layer, param_name, layer_name) do
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

    param_source = Enum.flat_map(pytorch_names, &param_refs(layer_name, &1))

    {[param_source], fn [value] -> value end}
  end

  defp param_refs(layer_name, param_name) do
    for layer_name <- List.wrap(layer_name) do
      {layer_name, param_name}
    end
  end

  defp lookup_param(pytorch_state, layer_name, pytorch_name) do
    # Note: the PyTorch model may have some root-level parameters that
    # we need to namespace under a layer in Axon, so after trying the
    # param within layer_name, we also try the param name directly
    pytorch_keys = [layer_name <> "." <> pytorch_name, pytorch_name]

    Enum.find_value(pytorch_keys, fn pytorch_key ->
      if value = pytorch_state[pytorch_key] do
        {value, pytorch_key}
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

  defp ensure_type(param_expr, value) do
    Utils.Nx.zip_with(param_expr, value, fn expr, tensor ->
      case {Nx.type(expr), Nx.type(tensor)} do
        {type, type} -> tensor
        {expected, _actual} -> Nx.as_type(tensor, expected)
      end
    end)
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
