defmodule Bumblebee.Text.QuestionAnswering do
  alias Bumblebee.Tokenizer
  alias Bumblebee.Shared
  alias Bumblebee.Utils
  alias Axon

  @moduledoc false

  def answer_question(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info

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

    {_init_fun, predict_fun} = Axon.build(model)

    Nx.Serving.new(
      fn ->
        predict_fun =
          Shared.compile_or_jit(predict_fun, defn_options, compile != nil, fn ->
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
          %Nx.Tensor{shape: {_}} = input ->
            {:ok, input}

          %{context: context, question: text} when is_binary(text) and is_binary(context) ->
            {:ok, {context, text}}

          other ->
            {:error, "expected a 1-dimensional tensor or {context,text}, got: #{inspect(other)}"}
        end)

      inputs = Bumblebee.apply_tokenizer(tokenizer, inputs)

      {Nx.Batch.concatenate([inputs]), {inputs, multi?}}
    end)
    |> Nx.Serving.client_postprocessing(fn outputs, metadata, {inputs, multi?} ->
      IO.inspect(inputs, label: "Given Inputs")
      Enum.zip_with(Utils.Nx.batch_to_list(inputs), Utils.Nx.batch_to_list(outputs), fn inputs,
                                                                                        outputs ->

        answer_start_index = outputs.start_logits |> Nx.argmax() |> Nx.to_number()
        answer_end_index = outputs.end_logits |> Nx.argmax() |> Nx.to_number()

        IO.inspect(Nx.to_flat_list(inputs["input_ids"]))

        answer_tokens =
          inputs["input_ids"][ answer_start_index..answer_end_index]

        answers = Bumblebee.Tokenizer.decode(tokenizer, answer_tokens)

        %{text: answers, start: answer_start_index, end: answer_end_index}
      end)
      |> Shared.normalize_output(multi?)
    end)
  end
end
