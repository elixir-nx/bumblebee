defmodule Bumblebee.Text.TextRerankingQwen3 do
  @moduledoc false

  alias Bumblebee.Shared

  def text_reranking_qwen3(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_causal_language_modeling)

    # Get yes/no token IDs
    yes_token =
      opts[:yes_token] ||
        get_token_id(tokenizer, "yes")

    no_token =
      opts[:no_token] ||
        get_token_id(tokenizer, "no")

    # Default Qwen3 reranker format
    instruction_prefix =
      opts[:instruction_prefix] ||
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"

    instruction_suffix =
      opts[:instruction_suffix] ||
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    task_description =
      opts[:task_description] ||
        "Given a web search query, retrieve relevant passages that answer the query"

    opts =
      Keyword.validate!(opts, [
        :compile,
        :yes_token,
        :no_token,
        :instruction_prefix,
        :instruction_suffix,
        :task_description,
        defn_options: [],
        preallocate_params: false
      ])

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
      Bumblebee.configure(tokenizer,
        length: sequence_length,
        return_token_type_ids: false
      )

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      outputs = predict_fun.(params, input)
      # outputs.logits has shape {batch_size, sequence_length, vocab_size}
      # Pool to last attended token position
      attention_mask = input["attention_mask"]

      sequence_lengths =
        attention_mask
        |> Nx.sum(axes: [1])
        |> Nx.subtract(1)
        |> Nx.as_type({:s, 64})

      last_token_logits = Bumblebee.Utils.Nx.batched_take(outputs.logits, sequence_lengths)

      # Extract logits for yes/no tokens
      yes_logits = last_token_logits[[.., yes_token]]
      no_logits = last_token_logits[[.., no_token]]

      stacked = Nx.stack([no_logits, yes_logits], axis: 1)
      log_probs = Axon.Activations.log_softmax(stacked, axis: 1)

      scores = Nx.exp(log_probs[[.., 1]])
      scores
    end

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        scope = {:scores, batch_key}

        scores_fun =
          Shared.compile_or_jit(scores_fun, scope, defn_options, compile != nil, fn ->
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
      {pairs, multi?} = validate_reranking_input!(input)

      # Format each query-document pair with the instruction template
      texts =
        Enum.map(pairs, fn {query, document} ->
          content = format_instruction(task_description, query, document)
          "#{instruction_prefix}#{content}#{instruction_suffix}"
        end)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts)
        end)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, {multi?, pairs}}
    end)
    |> Nx.Serving.client_postprocessing(fn {scores, _metadata}, {multi?, pairs} ->
      results =
        Enum.zip_with(Nx.to_list(scores), pairs, fn score, {query, document} ->
          %{score: score, query: query, document: document}
        end)

      output = %{scores: results}
      if multi?, do: output, else: %{scores: hd(results)}
    end)
  end

  defp format_instruction(task, query, document) do
    "<Instruct>: #{task}\n<Query>: #{query}\n<Document>: #{document}"
  end

  defp get_token_id(tokenizer, token) do
    encoded = Bumblebee.apply_tokenizer(tokenizer, token)
    Nx.to_flat_list(encoded["input_ids"]) |> hd()
  end

  defp validate_reranking_input!(input) do
    case input do
      {query, doc} when is_binary(query) and is_binary(doc) ->
        {[{query, doc}], false}

      list when is_list(list) ->
        pairs =
          Enum.map(list, fn
            {query, doc} when is_binary(query) and is_binary(doc) ->
              {query, doc}

            other ->
              raise ArgumentError,
                    "expected a query-document tuple {query, doc} where both are strings, got: #{inspect(other)}"
          end)

        {pairs, true}

      other ->
        raise ArgumentError,
              "expected a query-document tuple {query, doc} or a list of such tuples, got: #{inspect(other)}"
    end
  end
end
