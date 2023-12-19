defmodule Bumblebee.Text.FillMask do
  @moduledoc false

  alias Bumblebee.Shared

  def fill_mask(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_masked_language_modeling)

    opts =
      Keyword.validate!(opts, [:compile, top_k: 5, defn_options: [], preallocate_params: false])

    top_k = opts[:top_k]
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

    mask_token_id = Bumblebee.Tokenizer.special_token_id(tokenizer, :mask)
    mask_token = Bumblebee.Tokenizer.id_to_token(tokenizer, mask_token_id)

    tokenizer =
      Bumblebee.configure(tokenizer, length: sequence_length, return_token_type_ids: false)

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, inputs ->
      outputs = predict_fun.(params, inputs)
      scores = Axon.Activations.softmax(outputs.logits)

      mask_idx =
        inputs["input_ids"]
        |> Nx.equal(mask_token_id)
        |> Nx.argmax(axis: 1)

      {batch_size, _sequence_length, num_tokens} = Nx.shape(scores)

      mask_idx =
        mask_idx
        |> Nx.reshape({batch_size, 1, 1})
        |> Nx.broadcast({batch_size, 1, num_tokens})

      scores =
        scores
        |> Nx.take_along_axis(mask_idx, axis: 1)
        |> Nx.squeeze(axes: [1])

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

      texts = for text <- texts, do: validate_text!(text, mask_token)

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
            Enum.zip_with(top_scores, top_indices, fn score, token_id ->
              # Certain tokenizers distinguish tokens with a leading space,
              # so we normalize the result to a consistent string
              token =
                tokenizer
                |> Bumblebee.Tokenizer.decode([token_id])
                |> String.trim_leading()

              %{score: score, token: token}
            end)

          %{predictions: predictions}
        end
      )
      |> Shared.normalize_output(multi?)
    end)
  end

  defp validate_text!(text, mask_token) do
    mask_count = count_occurrences(text, "[MASK]")

    unless mask_count == 1 do
      raise ArgumentError,
            "expected exactly one occurrence of [MASK], got: #{mask_count} in #{inspect(text)}"
    end

    String.replace(text, "[MASK]", mask_token)
  end

  defp count_occurrences(string, substring) do
    string
    |> String.split(substring)
    |> length()
    |> Kernel.-(1)
  end
end
