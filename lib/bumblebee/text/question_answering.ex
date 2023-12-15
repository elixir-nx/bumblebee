defmodule Bumblebee.Text.QuestionAnswering do
  @moduledoc false

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  def question_answering(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_question_answering)

    opts = Keyword.validate!(opts, [:compile, defn_options: [], preallocate_params: false])

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
        return_token_type_ids: true,
        return_offsets: true
      )

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      outputs = predict_fun.(params, input)
      start_scores = Axon.Activations.softmax(outputs.start_logits)
      end_scores = Axon.Activations.softmax(outputs.end_logits)

      start_idx = Nx.argmax(start_scores, axis: -1, keep_axis: true)
      end_idx = Nx.argmax(end_scores, axis: -1, keep_axis: true)

      start_score = Nx.take_along_axis(start_scores, start_idx, axis: -1)
      end_score = Nx.take_along_axis(end_scores, end_idx, axis: -1)

      score = Nx.multiply(start_score, end_score)

      %{
        score: Nx.squeeze(score, axes: [-1]),
        start_idx: Nx.squeeze(start_idx, axes: [-1]),
        end_idx: Nx.squeeze(end_idx, axes: [-1])
      }
    end

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        predict_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32),
              "token_type_ids" => Nx.template({batch_size, sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          predict_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn raw_input ->
      {raw_inputs, multi?} =
        Shared.validate_serving_input!(raw_input, fn
          %{question: question, context: context}
          when is_binary(question) and is_binary(context) ->
            {:ok, {question, context}}

          other ->
            {:error,
             "expected input map with :question and :context keys, got: #{inspect(other)}"}
        end)

      all_inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, raw_inputs)
        end)

      inputs = Map.take(all_inputs, ["input_ids", "attention_mask", "token_type_ids"])

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, {all_inputs, raw_inputs, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn {outputs, _metadata}, {inputs, raw_inputs, multi?} ->
      Enum.zip_with(
        [raw_inputs, Utils.Nx.batch_to_list(inputs), Utils.Nx.batch_to_list(outputs)],
        fn [{_question_text, context_text}, inputs, outputs] ->
          start_idx = Nx.to_number(outputs.start_idx)
          end_idx = Nx.to_number(outputs.end_idx)
          score = Nx.to_number(outputs.score)
          start = Nx.to_number(inputs["start_offsets"][start_idx])
          ending = Nx.to_number(inputs["end_offsets"][end_idx])

          answer_text = binary_part(context_text, start, ending - start)

          results = [
            %{text: answer_text, start: start, end: ending, score: score}
          ]

          %{results: results}
        end
      )
      |> Shared.normalize_output(multi?)
    end)
  end
end
