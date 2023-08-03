defmodule Bumblebee.Text.FillMask do
  @moduledoc false

  alias Bumblebee.Shared

  def fill_mask(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_masked_language_modeling)
    opts = Keyword.validate!(opts, [:compile, top_k: 5, defn_options: []])

    top_k = opts[:top_k]
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

      scores
      |> Nx.take_along_axis(mask_idx, axis: 1)
      |> Nx.squeeze(axes: [1])
    end

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
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
          scores_fun.(params, inputs)
        end
      end,
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size, batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      texts = for text <- texts, do: validate_text!(text, mask_token)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts,
            length: sequence_length,
            return_token_type_ids: false
          )
        end)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {scores, _metadata}, multi? ->
      for scores <- Bumblebee.Utils.Nx.batch_to_list(scores) do
        k = min(top_k, Nx.size(scores))
        {top_scores, top_indices} = Nx.top_k(scores, k: k)

        predictions =
          Enum.zip_with(
            Nx.to_flat_list(top_scores),
            Nx.to_flat_list(top_indices),
            fn score, token_id ->
              # Certain tokenizers distinguish tokens with a leading space,
              # so we normalize the result to a consistent string
              token =
                tokenizer
                |> Bumblebee.Tokenizer.decode([token_id])
                |> String.trim_leading()

              %{score: score, token: token}
            end
          )

        %{predictions: predictions}
      end
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
