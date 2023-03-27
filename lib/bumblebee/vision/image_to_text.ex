defmodule Bumblebee.Vision.ImageToText do
  @moduledoc false

  alias Bumblebee.Shared

  def image_to_text(model_info, featurizer, tokenizer, opts \\ []) do
    {compile, opts} = Keyword.pop(opts, :compile)
    {defn_options, opts} = Keyword.pop(opts, :defn_options, [])

    batch_size = compile[:batch_size]

    if compile != nil and batch_size == nil do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size, got: #{inspect(compile)}"
    end

    %{model: model, params: params, spec: spec} = model_info

    generate_fun = Bumblebee.Text.Generation.build_generate(model, spec, opts)

    Nx.Serving.new(
      fn defn_options ->
        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "pixel_values" => Shared.input_template(spec, "pixel_values", [batch_size])
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          generate_fun.(params, inputs)
        end
      end,
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {images, multi?} = Shared.validate_serving_input!(input, &Shared.validate_image/1)
      inputs = Bumblebee.apply_featurizer(featurizer, images, defn_options: defn_options)
      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn token_ids, _metadata, multi? ->
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      decoded
      |> Enum.map(&%{results: [%{text: &1}]})
      |> Shared.normalize_output(multi?)
    end)
  end
end
