defmodule Bumblebee.Vision.ImageToText do
  @moduledoc false

  alias Bumblebee.Shared
  alias Bumblebee.Text

  def image_to_text(
        model_info,
        featurizer,
        tokenizer,
        %Text.GenerationConfig{} = generation_config,
        opts \\ []
      ) do
    opts = Keyword.validate!(opts, [:compile, defn_options: [], preallocate_params: false])

    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [:for_conditional_generation])

    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size])
        |> Shared.require_options!([:batch_size])
      end

    batch_size = compile[:batch_size]

    generate_fun = Text.Generation.build_generate(model, spec, generation_config)

    generate_fun = fn params, {inputs, seed} ->
      inputs = Bumblebee.Featurizer.process_batch(featurizer, inputs)
      inputs = Map.put(inputs, "seed", seed)
      %{token_ids: token_ids} = generate_fun.(params, inputs)
      token_ids
    end

    Nx.Serving.new(
      fn defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            inputs = Bumblebee.Featurizer.batch_template(featurizer, batch_size)
            seed = Nx.template({batch_size}, :s64)
            [params, {inputs, seed}]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          generate_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = Shared.validate_serving_input!(input, &validate_input/1)

      images = Enum.map(inputs, & &1.image)
      seed = Enum.map(inputs, & &1.seed) |> Nx.tensor(backend: Nx.BinaryBackend)

      inputs = Bumblebee.Featurizer.process_input(featurizer, images)
      {Nx.Batch.concatenate([{inputs, seed}]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {token_ids, _metadata}, multi? ->
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      decoded
      |> Enum.map(&%{results: [%{text: &1}]})
      |> Shared.normalize_output(multi?)
    end)
  end

  defp validate_input(%{image: image} = input) do
    if Shared.image?(image) do
      {:ok, %{image: image, seed: input[:seed] || :erlang.system_time()}}
    else
      {:error, "expected an image, got: #{inspect(image)}"}
    end
  end

  defp validate_input(input), do: validate_input(%{image: input})
end
