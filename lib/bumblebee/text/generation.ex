defmodule Bumblebee.Text.Generation do
  @moduledoc """
  Utilities for sequence generation using language models.
  """

  @doc """
  Initializes an opaque cache input for iterative inference.
  """
  @callback init_cache(
              config :: Bumblebee.ModelSpec.t(),
              batch_size :: pos_integer(),
              max_length :: pos_integer(),
              inputs :: map()
            ) :: Nx.t()

  import Nx.Defn

  @doc """
  Initializes an opaque cache input for iterative inference.
  """
  @spec init_cache(Bumblebee.ModelSpec.t(), pos_integer(), pos_integer(), map()) :: Nx.t()
  def init_cache(%module{} = config, batch_size, max_length, inputs) do
    module.init_cache(config, batch_size, max_length, inputs)
  end

  @doc """
  Generates sequences of token ids using the given language model.

  The model should be either a decoder or an encoder-decoder. The tokens
  are generated by iterative inference using the decoder (autoregression),
  until the termination criteria are met.

  In case of encoder-decoder models, the corresponding encoder is run
  only once and the intermediate state is reused during all iterations.

  The length of the generated sequence is not fixed, however it can be
  controlled via several options.

  ## Options

    * `:max_length` - the maximum length of the sequence to be generated

    * `:min_length` - the minimum length of the sequence to be generated

    * `:decoder_start_token_id` - the id of the initial token when
      generating from scratch, in case of encoder-decoder models

    * `:bos_token_id` - the id of the beginning-of-sequence token

    * `:eos_token_id` - the id of the end-of-sequence token

    * `:pad_token_id` - the id of the padding token

    * `:forced_bos_token_id` - the id of the token to force as the first
      generated token

    * `:forced_eos_token_id` - the id of the token to force as the last
      generated token when `:max_length` is reached

  The default option values are taken from the given model configuration
  when available.
  """
  @spec generate(Bumblebee.ModelSpec.t(), Axon.t(), map(), map(), keyword()) :: Nx.t()
  def generate(config, model, params, inputs, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        max_length: Map.get(config, :max_length),
        min_length: Map.get(config, :min_length),
        decoder_start_token_id: Map.get(config, :decoder_start_token_id),
        bos_token_id: Map.get(config, :bos_token_id),
        eos_token_id: Map.get(config, :eos_token_id),
        pad_token_id: Map.get(config, :pad_token_id),
        forced_bos_token_id: Map.get(config, :forced_bos_token_id),
        forced_eos_token_id: Map.get(config, :forced_eos_token_id)
      )

    max_length = opts[:max_length]
    min_length = opts[:min_length]
    decoder_start_token_id = opts[:decoder_start_token_id] || opts[:bos_token_id]
    eos_token_id = opts[:eos_token_id]
    pad_token_id = opts[:pad_token_id]
    forced_bos_token_id = opts[:forced_bos_token_id]
    forced_eos_token_id = opts[:forced_eos_token_id]

    {prepare_inputs_fun, update_inputs_fun} =
      input_callbacks(config, model, max_length, decoder_start_token_id)

    {_init_fun, predict_fun} = Axon.build(model)

    logits_processor_fun =
      get_logits_processor(
        min_length,
        max_length,
        eos_token_id,
        forced_bos_token_id,
        forced_eos_token_id
      )

    generate(
      inputs,
      predict_fun,
      params,
      logits_processor_fun,
      prepare_inputs_fun,
      update_inputs_fun,
      max_length: max_length,
      pad_token_id: pad_token_id,
      eos_token_id: eos_token_id
    )
  end

  defp encoder_from_encoder_decoder(model) do
    # We cherry-pick encoder output from the encoder-decoder output.
    # The expanded expression will have no decoder bits, so it will
    # effectively be the same as an encoder built from scratch

    Axon.nx(model, fn output ->
      case output do
        %{
          encoder_last_hidden_state: last_hidden_state,
          encoder_hidden_states: hidden_states,
          encoder_attentions: attentions
        } ->
          %{
            last_hidden_state: last_hidden_state,
            hidden_states: hidden_states,
            attentions: attentions
          }

        _ ->
          raise ArgumentError,
                "expected an encoder-decoder model, but it does not have the expected output"
      end
    end)
  end

  defp input_callbacks(config, model, max_length, decoder_start_token_id) do
    if encoder_decoder?(model) do
      encoder = encoder_from_encoder_decoder(model)
      {_encoder_init_fun, encoder_predict_fun} = Axon.build(encoder)

      prepare_inputs_fun = fn inputs, params ->
        encoder_output = encoder_predict_fun.(params, inputs)

        batch_size = Nx.axis_size(inputs["input_ids"], 0)
        decoder_input_ids = Nx.broadcast(decoder_start_token_id, {batch_size, 1})

        inputs =
          Map.merge(inputs, %{
            "encoder_last_hidden_state" => encoder_output.last_hidden_state,
            "decoder_input_ids" => decoder_input_ids
          })

        inputs = prepare_decoder_inputs(inputs, "decoder_", config, max_length)
        {inputs, inputs["decoder_input_ids"]}
      end

      update_inputs_fun = &update_decoder_inputs(&1, &2, &3, "decoder_")

      {prepare_inputs_fun, update_inputs_fun}
    else
      prepare_inputs_fun = fn inputs, _params ->
        inputs = prepare_decoder_inputs(inputs, "", config, max_length)
        {inputs, inputs["input_ids"]}
      end

      update_inputs_fun = &update_decoder_inputs(&1, &2, &3, "")

      {prepare_inputs_fun, update_inputs_fun}
    end
  end

  defp encoder_decoder?(model) do
    inputs = Axon.get_inputs(model)
    Map.has_key?(inputs, "input_ids") and Map.has_key?(inputs, "decoder_input_ids")
  end

  defp prepare_decoder_inputs(inputs, prefix, config, max_length) do
    input_ids = inputs[prefix <> "input_ids"]
    attention_mask = inputs[prefix <> "attention_mask"] || Nx.broadcast(1.0, input_ids)

    position_ids =
      attention_mask
      |> Nx.cumulative_sum(axis: 1)
      |> Nx.subtract(1)

    inputs =
      inputs
      |> Map.put(prefix <> "attention_mask", attention_mask)
      |> Map.put(prefix <> "position_ids", position_ids)

    batch_size = Nx.axis_size(input_ids, 0)
    cache = init_cache(config, batch_size, max_length, inputs)
    Map.put(inputs, "cache", cache)
  end

  defp update_decoder_inputs(inputs, output, token_ids, prefix) do
    inputs
    |> Map.replace!(prefix <> "input_ids", token_ids)
    |> Map.replace!(prefix <> "attention_mask", Nx.broadcast(1.0, token_ids))
    |> Map.update!(prefix <> "position_ids", fn position_ids ->
      position_ids
      |> Nx.slice_along_axis(Nx.axis_size(position_ids, -1) - 1, 1, axis: -1)
      |> Nx.add(1)
    end)
    |> Map.replace!("cache", output.cache)
  end

  defp get_logits_processor(
         min_length,
         max_length,
         eos_token_id,
         forced_bos_token_id,
         forced_eos_token_id
       ) do
    processors = [
      if min_length && min_length > 0 && eos_token_id do
        &min_length_logits_processor(&1, &2, &3,
          min_length: min_length,
          eos_token_id: eos_token_id
        )
      end,
      if forced_bos_token_id do
        &bos_token_logits_processor(&1, &2, &3, bos_token_id: forced_bos_token_id)
      end,
      if forced_eos_token_id do
        &eos_token_logits_processor(&1, &2, &3,
          max_length: max_length,
          eos_token_id: forced_eos_token_id
        )
      end
    ]

    fn logits, sequences, length ->
      for processor <- processors, processor, reduce: logits do
        logits -> processor.(logits, sequences, length)
      end
    end
  end

  defnp generate(
          inputs,
          predict_fun,
          params,
          logits_processor_fun,
          prepare_inputs_fun,
          update_inputs_fun,
          opts \\ []
        ) do
    {decoder_inputs, decoder_input_ids} = prepare_inputs_fun.(inputs, params)

    greedy(
      decoder_inputs,
      decoder_input_ids,
      predict_fun,
      params,
      logits_processor_fun,
      update_inputs_fun,
      opts
    )
  end

  defnp greedy(
          inputs,
          decoder_input_ids,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          opts \\ []
        ) do
    max_length = opts[:max_length]
    pad_token_id = opts[:pad_token_id]
    eos_token_id = opts[:eos_token_id]

    {batch_size, length} = Nx.shape(decoder_input_ids)

    sequences = Nx.broadcast(pad_token_id, {batch_size, max_length})
    sequences = Nx.put_slice(sequences, [0, 0], decoder_input_ids)

    finished? = Nx.broadcast(Nx.tensor(0, type: :u8), {batch_size})

    # The loop works with inputs of length 1, so if the initial input
    # is longer, we make the initial pass outside
    {sequences, length, finished?, inputs} =
      if length > 1 do
        greedy_step(
          sequences,
          length,
          finished?,
          inputs,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          pad_token_id: pad_token_id,
          eos_token_id: eos_token_id
        )
      else
        {sequences, length, finished?, inputs}
      end

    {sequences, _length, _finished?, _inputs, _params} =
      while {sequences, length, finished?, inputs, params},
            greedy_condition(finished?, length, max_length) do
        {sequences, length, finished?, inputs} =
          greedy_step(
            sequences,
            length,
            finished?,
            inputs,
            predict_fun,
            params,
            logits_processor_fun,
            update_inputs_fun,
            pad_token_id: pad_token_id,
            eos_token_id: eos_token_id
          )

        {sequences, length, finished?, inputs, params}
      end

    sequences
  end

  defnp greedy_condition(finished?, length, max_length) do
    not Nx.all(finished?) and length < max_length
  end

  defnp greedy_step(
          sequences,
          length,
          finished?,
          inputs,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          opts
        ) do
    pad_token_id = opts[:pad_token_id]
    eos_token_id = opts[:eos_token_id]

    output = predict_fun.(params, inputs)

    logits = output.logits[[0..-1//1, -1]]
    logits = logits_processor_fun.(logits, sequences, length)

    token_id = Nx.argmax(logits, axis: -1)

    token_id = Nx.select(finished?, pad_token_id, token_id)

    finished? =
      transform({finished?, eos_token_id}, fn
        {finished?, nil} -> finished?
        {finished?, eos_token_id} -> finished? or token_id == eos_token_id
      end)

    token_id = Nx.new_axis(token_id, -1)

    sequences = Nx.put_slice(sequences, [0, length], token_id)

    inputs = update_inputs_fun.(inputs, output, token_id)

    {sequences, length + 1, finished?, inputs}
  end

  # Logit processors

  defnp bos_token_logits_processor(logits, sequences, length, opts \\ []) do
    opts = keyword!(opts, [:bos_token_id])
    bos_token_id = opts[:bos_token_id]

    if length == 1 do
      force_token_id(logits, sequences, token_id: bos_token_id)
    else
      logits
    end
  end

  defnp eos_token_logits_processor(logits, sequences, length, opts \\ []) do
    opts = keyword!(opts, [:eos_token_id, :max_length])
    eos_token_id = opts[:eos_token_id]
    max_length = opts[:max_length]

    if length == max_length - 1 do
      force_token_id(logits, sequences, token_id: eos_token_id)
    else
      logits
    end
  end

  defnp min_length_logits_processor(logits, sequences, length, opts \\ []) do
    opts = keyword!(opts, [:eos_token_id, :min_length])
    eos_token_id = opts[:eos_token_id]
    min_length = opts[:min_length]

    if length < min_length do
      ignore_token_id(logits, sequences, token_id: eos_token_id)
    else
      logits
    end
  end

  defnp force_token_id(logits, sequences, opts \\ []) do
    token_id = opts[:token_id]

    batch_size = Nx.axis_size(sequences, 0)

    Nx.Constants.neg_infinity()
    |> Nx.broadcast(logits)
    |> Nx.put_slice([0, token_id], Nx.broadcast(0, {batch_size, 1}))
  end

  defnp ignore_token_id(logits, sequences, opts \\ []) do
    token_id = opts[:token_id]

    batch_size = Nx.axis_size(sequences, 0)

    Nx.put_slice(
      logits,
      [0, token_id],
      Nx.broadcast(Nx.Constants.neg_infinity(), {batch_size, 1})
    )
  end
end
