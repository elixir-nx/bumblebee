defmodule Bumblebee.Text.QuestionAnswering do
  alias Bumblebee.Shared
  alias Bumblebee.Utils
  alias Axon

  @moduledoc false

  def question_answering(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_question_answering)

    opts =
      Keyword.validate!(opts, [
        :compile,
        doc_stride: 128,
        top_k: 1,
        defn_options: []
      ])

    top_k = opts[:top_k]
    compile = opts[:compile]
    defn_options = opts[:defn_options]
    doc_stride = opts[:doc_stride]

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    {_init_fun, predict_fun} = Axon.build(model)

    scores_fun = fn params, input ->
       predict_fun.(params, input)
    end

    Nx.Serving.new(
      fn ->
        predict_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :s64),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :s64)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)

          predict_fun.(params, inputs)
        end
      end,
      batch_size: batch_size
    )
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} =
        Shared.validate_serving_input!(input, fn
          %{question: question, context: context}
          when is_binary(question) and is_binary(context) ->
            {:ok, {question, context}}

          other ->
            {:error,
             "expected input map with :question and :context keys, got: #{inspect(other)}"}
        end)

      all_inputs =
        Bumblebee.apply_tokenizer(tokenizer, inputs,
          length: sequence_length,
          return_special_tokens_mask: true,
          return_offsets: true
        )

      inputs = Map.take(all_inputs, ["input_ids", "attention_mask"])
      {Nx.Batch.concatenate([inputs]), {Map.merge(all_inputs, %{context_info: input}), multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn outputs, _metadata, {inputs, _multi?} ->

       %{context_info: %{context: context}} = inputs
      %{
        results:
          Enum.zip_with(
            Utils.Nx.batch_to_list(Map.drop(inputs, [:context_info])),
            Utils.Nx.batch_to_list(outputs),
            fn inputs, outputs ->
              start_scores = Axon.Activations.softmax(outputs.start_logits)
              start_idx = start_scores |> Nx.argmax() |> Nx.to_number()

              end_scores = Axon.Activations.softmax(outputs.end_logits)
              end_idx = end_scores |> Nx.argmax() |> Nx.to_number()
              start = inputs["start_offsets"][start_idx] |> Nx.to_number()
              ending = inputs["end_offsets"][end_idx] |> Nx.to_number()

              score = Nx.multiply(start_scores[start_idx], end_scores[end_idx]) |> Nx.to_number()

              answer = binary_part(context, start, ending - start )

              %{
                text: answer,
                start: start,
                end: ending,
                score: score
              }
            end
          )
      }
    end)
  end
end
