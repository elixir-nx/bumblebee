defmodule Bumblebee.Text.TextClassification do
  @moduledoc false

  alias Bumblebee.Shared

  def text_classification(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_sequence_classification)

    opts =
      Keyword.validate!(opts, [
        :compile,
        top_k: 5,
        scores_function: :softmax,
        defn_options: [],
        preallocate_params: false
      ])

    top_k = opts[:top_k]
    scores_function = opts[:scores_function]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size, :sequence_length])
        |> Shared.require_options!([:batch_size, :sequence_length])
      end

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    tokenizer =
      Bumblebee.configure(tokenizer, length: sequence_length, return_token_type_ids: false)

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      outputs = predict_fun.(params, input)
      scores = Shared.logits_to_scores(outputs.logits, scores_function)
      k = min(top_k, Nx.axis_size(scores, 1))
      {top_scores, top_indices} = Nx.top_k(scores, k: k)
      {top_scores, top_indices}
    end

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        scores_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          scores_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts)
        end)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {{top_scores, top_indices}, _metadata}, multi? ->
      Enum.zip_with(
        Nx.to_list(top_scores),
        Nx.to_list(top_indices),
        fn top_scores, top_indices ->
          predictions =
            Enum.zip_with(top_scores, top_indices, fn score, idx ->
              label = spec.id_to_label[idx] || "LABEL_#{idx}"
              %{score: score, label: label}
            end)

          %{predictions: predictions}
        end
      )
      |> Shared.normalize_output(multi?)
    end)
  end
end
