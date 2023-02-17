defmodule Bumblebee.Text.QuestionAnswering do
  @moduledoc false

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  def question_answering(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_question_answering)

    opts = Keyword.validate!(opts, [:compile, defn_options: []])

    compile = opts[:compile]
    defn_options = opts[:defn_options]

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
      outputs = predict_fun.(params, input)
      start_scores = Axon.Activations.softmax(outputs.start_logits)
      end_scores = Axon.Activations.softmax(outputs.end_logits)
      %{start_scores: start_scores, end_scores: end_scores}
    end

    Nx.Serving.new(
      fn defn_options ->
        predict_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :s64),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :s64),
              "token_type_ids" => Nx.template({batch_size, sequence_length}, :s64)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)

          predict_fun.(params, inputs)
        end
      end,
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size)
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
        Bumblebee.apply_tokenizer(tokenizer, raw_inputs,
          length: sequence_length,
          return_token_type_ids: true,
          return_offsets: true
        )

      inputs = Map.take(all_inputs, ["input_ids", "attention_mask", "token_type_ids"])
      {Nx.Batch.concatenate([inputs]), {all_inputs, raw_inputs, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn outputs, _metadata, {inputs, raw_inputs, multi?} ->
      Enum.zip_with(
        [raw_inputs, Utils.Nx.batch_to_list(inputs), Utils.Nx.batch_to_list(outputs)],
        fn [{_question_text, context_text}, inputs, outputs] ->
          start_idx = outputs.start_scores |> Nx.argmax() |> Nx.to_number()
          end_idx = outputs.end_scores |> Nx.argmax() |> Nx.to_number()

          start = Nx.to_number(inputs["start_offsets"][start_idx])
          ending = Nx.to_number(inputs["end_offsets"][end_idx])

          score =
            outputs.start_scores[start_idx]
            |> Nx.multiply(outputs.end_scores[end_idx])
            |> Nx.to_number()

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
