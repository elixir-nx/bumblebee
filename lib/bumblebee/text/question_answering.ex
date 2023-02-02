defmodule Bumblebee.Text.QuestionAnswering do
  alias Bumblebee.Tokenizer
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
      # input = Utils.Nx.composite_flatten_batch(input)
      output = predict_fun.(params, input)
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

          scores = predict_fun.(params, inputs)
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
      {Nx.Batch.concatenate([inputs]), {all_inputs, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn outputs, metadata, {inputs, multi?} ->
      %{
        results:
          Enum.zip_with(
            Utils.Nx.batch_to_list(inputs),
            Utils.Nx.batch_to_list(outputs),
            fn inputs, outputs ->
              answer_start_index = outputs.start_logits |> Nx.argmax() |> Nx.to_number()

              answer_end_index = outputs.end_logits |> Nx.argmax() |> Nx.to_number()


              start = inputs["start_offsets"][answer_start_index]
              ending = inputs["end_offsets"][answer_end_index]
              answer_tokens = inputs["input_ids"][answer_start_index..answer_end_index]

              answers = Bumblebee.Tokenizer.decode(tokenizer, answer_tokens)

              %{
                text: answers,
                start: start,
                end: ending,
                score: 0
              }
            end
          )
      }
    end)
  end

  defp gather_raw_entities(scores, sequence_length, tokenizer, inputs) do
    flat_special_tokens_mask = Nx.to_flat_list(inputs["special_tokens_mask"])
    flat_input_ids = Nx.to_flat_list(inputs["input_ids"])
    flat_start_offsets = Nx.to_flat_list(inputs["start_offsets"])
    flat_end_offsets = Nx.to_flat_list(inputs["end_offsets"])

    # TODO: Optional offset mapping
    # TODO: Non-BPE tokenizers
    IO.inspect(sequence_length, label: "Sequence Length")

    token_infos =
      Enum.zip([
        0..(sequence_length - 1),
        flat_input_ids,
        flat_start_offsets,
        flat_end_offsets,
        flat_special_tokens_mask
      ])

    for {token_idx, token_id, start_idx, end_idx, _special? = 0} <- token_infos do
      token = Bumblebee.Tokenizer.id_to_token(tokenizer, token_id)
      # Indices are expressed in terms of utf8 bytes
      token_reference_length = end_idx - start_idx

      token_scores = scores[token_idx]

      %{
        token: token,
        token_id: token_id,
        scores: token_scores,
        start: start_idx,
        end: end_idx,
        index: token_idx,
        # Subword tokens usually have the ## prefix, so they are longer
        # than the actual word piece
        is_subword: byte_size(token) != token_reference_length
      }
    end
  end

  defp aggregate(entities, spec, _tokenizer, nil) do
    entities
    |> add_token_labels(spec)
    |> Enum.map(fn entity ->
      %{
        start: entity.start,
        end: entity.end,
        label: entity.label,
        score: entity.score,
        phrase: entity.token
      }
    end)
  end

  defp add_token_labels(entities, spec) do
    Enum.map(entities, fn entity ->
      entity_idx = entity.scores |> Nx.argmax() |> Nx.to_number()
      score = Nx.to_number(entity.scores[entity_idx])
      label = spec.id_to_label[entity_idx]
      Map.merge(entity, %{label: label, score: score})
    end)
  end
end
