defmodule Bumblebee.Text.Generation do
  @moduledoc """
  An interface for language models supporting sequence generation.
  """

  @type cache :: Nx.Tensor.t() | Nx.Container.t()

  @doc """
  Initializes an opaque cache input for iterative inference.
  """
  @callback init_cache(
              spec :: Bumblebee.ModelSpec.t(),
              batch_size :: pos_integer(),
              max_length :: pos_integer(),
              inputs :: map()
            ) :: cache()

  @doc """
  Traverses all batched tensors in the cache.

  This function is used when the cache needs to be inflated or
  deflated for a different batch size.
  """
  @callback traverse_cache(
              spec :: Bumblebee.ModelSpec.t(),
              cache(),
              (Nx.Tensor.t() -> Nx.Tensor.t())
            ) :: cache()

  @doc """
  Returns a configuration module for extra model-specific generation
  attributes to extend the base `Bumblebee.Text.GenerationConfig`.
  """
  @callback extra_config_module(spec :: Bumblebee.ModelSpec.t()) :: module()

  @optional_callbacks extra_config_module: 1

  import Nx.Defn

  alias Bumblebee.Shared
  alias Bumblebee.Utils
  alias Bumblebee.Text

  @doc """
  Initializes an opaque cache input for iterative inference.
  """
  @spec init_cache(Bumblebee.ModelSpec.t(), pos_integer(), pos_integer(), map()) :: cache()
  def init_cache(%module{} = spec, batch_size, max_length, inputs) do
    module.init_cache(spec, batch_size, max_length, inputs)
  end

  @doc """
  Calls `fun` for every batched tensor in the cache.
  """
  @spec traverse_cache(
          Bumblebee.ModelSpec.t(),
          cache,
          (Nx.Tensor.t() -> Nx.Tensor.t())
        ) :: cache()
  def traverse_cache(%module{} = spec, cache, fun) do
    module.traverse_cache(spec, cache, fun)
  end

  @doc """
  Returns a configuration module for extra model-specific generation
  attributes to extend the base `Bumblebee.Text.GenerationConfig`.
  """
  @spec extra_config_module(Bumblebee.ModelSpec.t()) :: module() | nil
  def extra_config_module(%module{} = spec) do
    if Code.ensure_loaded?(module) and function_exported?(module, :extra_config_module, 1) do
      module.extra_config_module(spec)
    end
  end

  @doc """
  Builds a numerical definition that generates sequences of tokens using
  the given language model.

  The model should be either a decoder or an encoder-decoder. The tokens
  are generated by iterative inference using the decoder (autoregression),
  until the termination criteria are met.

  In case of encoder-decoder models, the corresponding encoder is run
  only once and the intermediate state is reused during all iterations.

  The generation is controlled by a number of options given as
  `%Bumblebee.Text.GenerationConfig{}`, see the corresponding docs
  for more details.

  ## Options

    * `:seed` - random seed to use when sampling. By default the current
      timestamp is used

    * `:logits_processors` - a list of numerical functions to modify
      predicted scores at each generation step. The functions are
      applied in order, after all default processors

  """
  @spec build_generate(
          Axon.t(),
          Bumblebee.ModelSpec.t(),
          Bumblebee.Text.GenerationConfig.t(),
          keyword()
        ) :: (params :: map(), inputs :: map() -> Nx.t())
  def build_generate(model, spec, config, opts \\ []) do
    opts = Keyword.validate!(opts, [:seed, logits_processors: []])
    seed = Keyword.get_lazy(opts, :seed, &:erlang.system_time/0)

    decoder_start_token_id = config.decoder_start_token_id || config.bos_token_id
    eos_token_id = config.eos_token_id
    pad_token_id = config.pad_token_id || config.eos_token_id

    {max_length_fun, min_length_fun} = lazy_lengths_from_opts(config)

    {prepare_inputs_fun, update_inputs_fun} =
      input_callbacks(model, spec, max_length_fun, decoder_start_token_id)

    traverse_cache_fun = &traverse_cache(spec, &1, &2)

    model =
      if not spec.output_hidden_states and config.strategy.type == :contrastive_search do
        spec
        |> Bumblebee.configure(output_hidden_states: true)
        |> Bumblebee.build_model()
      else
        model
      end

    {_init_fun, predict_fun} = Axon.build(model)

    logits_processor_fun = get_logits_processor(min_length_fun, config, opts[:logits_processors])

    &generate_impl(
      &2,
      predict_fun,
      &1,
      logits_processor_fun,
      prepare_inputs_fun,
      update_inputs_fun,
      traverse_cache_fun,
      pad_token_id: pad_token_id,
      eos_token_id: eos_token_id,
      seed: seed,
      strategy: config.strategy
    )
  end

  defp lazy_lengths_from_opts(opts) do
    max_length_fun =
      case {opts.max_new_tokens, opts.max_length} do
        {nil, nil} ->
          raise ArgumentError,
                "expected either :max_new_tokens or :max_length option, but neither was given"

        {max_new_tokens, nil} ->
          fn input_length -> input_length + max_new_tokens end

        {nil, max_length} ->
          fn _ -> max_length end
      end

    min_length_fun =
      case {opts.min_new_tokens, opts.min_length} do
        {nil, nil} ->
          nil

        {min_new_tokens, nil} ->
          fn input_length -> input_length + min_new_tokens end

        {nil, min_length} ->
          fn _ -> min_length end
      end

    {max_length_fun, min_length_fun}
  end

  defp encoder_from_encoder_decoder(model) do
    # We cherry-pick encoder outputs from the encoder-decoder outputs.
    # The expanded expression will have no decoder bits, so it will
    # effectively be the same as an encoder built from scratch

    Axon.nx(model, fn outputs ->
      case outputs do
        %{
          encoder_hidden_state: hidden_state,
          encoder_hidden_states: hidden_states,
          encoder_attentions: attentions
        } ->
          %{
            hidden_state: hidden_state,
            hidden_states: hidden_states,
            attentions: attentions
          }

        _ ->
          raise ArgumentError,
                "expected an encoder-decoder model, but it does not have the expected outputs"
      end
    end)
  end

  defp input_callbacks(model, spec, max_length_fun, decoder_start_token_id) do
    if encoder_decoder?(model) do
      encoder = encoder_from_encoder_decoder(model)
      {_encoder_init_fun, encoder_predict_fun} = Axon.build(encoder)

      prepare_inputs_fun = fn inputs, params ->
        encoder_outputs = encoder_predict_fun.(params, inputs)

        batch_size = Nx.axis_size(encoder_input(inputs), 0)
        decoder_input_ids = Nx.broadcast(decoder_start_token_id, {batch_size, 1})

        inputs =
          Map.merge(inputs, %{
            "encoder_hidden_state" => encoder_outputs.hidden_state,
            "decoder_input_ids" => decoder_input_ids
          })

        max_length = max_length_fun.(1)
        inputs = prepare_decoder_inputs(inputs, "decoder_", spec, max_length)
        {inputs, inputs["decoder_input_ids"], max_length}
      end

      update_inputs_fun = &update_decoder_inputs("decoder_", &1, &2, &3)

      {prepare_inputs_fun, update_inputs_fun}
    else
      prepare_inputs_fun = fn inputs, _params ->
        sequence_length = Nx.axis_size(inputs["input_ids"], 1)
        max_length = max_length_fun.(sequence_length)
        inputs = prepare_decoder_inputs(inputs, "", spec, max_length)
        {inputs, inputs["input_ids"], max_length}
      end

      update_inputs_fun = &update_decoder_inputs("", &1, &2, &3)

      {prepare_inputs_fun, update_inputs_fun}
    end
  end

  defp encoder_decoder?(model) do
    inputs = Axon.get_inputs(model)
    encoder_input(inputs) != nil and Map.has_key?(inputs, "decoder_input_ids")
  end

  defp encoder_input(inputs) do
    inputs["input_ids"] || inputs["input_features"] || inputs["pixel_values"]
  end

  defp prepare_decoder_inputs(inputs, prefix, spec, max_length) do
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
    cache = init_cache(spec, batch_size, max_length, inputs)
    Map.put(inputs, "cache", cache)
  end

  defp update_decoder_inputs(prefix, inputs, cache, token_ids) do
    inputs
    |> Map.replace!(prefix <> "input_ids", token_ids)
    |> Map.replace!(prefix <> "attention_mask", Nx.broadcast(1.0, token_ids))
    |> Map.update!(prefix <> "position_ids", fn position_ids ->
      position_ids
      |> Nx.slice_along_axis(Nx.axis_size(position_ids, -1) - 1, 1, axis: -1)
      |> Nx.add(1)
    end)
    |> Map.replace!("cache", cache)
  end

  defp get_logits_processor(min_length_fun, config, logits_processors) do
    import Bumblebee.Text.Generation.LogitsProcessing

    processors =
      [
        if config.no_repeat_ngram_length && config.no_repeat_ngram_length > 0 do
          &no_repeat_ngram_processor(&1, &2, ngram_length: config.no_repeat_ngram_length)
        end,
        if min_length_fun && config.eos_token_id do
          &min_length_processor(&1, &2,
            min_length_fun: min_length_fun,
            eos_token_id: config.eos_token_id
          )
        end,
        if config.suppressed_token_ids != [] do
          &suppressed_tokens_processor(&1, &2, suppressed_token_ids: config.suppressed_token_ids)
        end,
        if config.forced_bos_token_id do
          &bos_token_processor(&1, &2, bos_token_id: config.forced_bos_token_id)
        end,
        if config.forced_eos_token_id do
          &eos_token_processor(&1, &2, eos_token_id: config.forced_eos_token_id)
        end,
        if config.forced_token_ids do
          &forced_tokens_processor(&1, &2, forced_token_ids: config.forced_token_ids)
        end
      ] ++
        if config.strategy.type == :multinomial_sampling do
          [
            if top_k = config.strategy[:top_k] do
              &top_k_processor(&1, &2, top_k: top_k)
            end,
            if top_p = config.strategy[:top_p] do
              &top_p_processor(&1, &2, top_p: top_p)
            end
          ]
        else
          []
        end ++ logits_processors

    fn logits, context ->
      for processor <- processors, processor, reduce: logits do
        logits -> processor.(logits, context)
      end
    end
  end

  deftransformp generate_impl(
                  inputs,
                  predict_fun,
                  params,
                  logits_processor_fun,
                  prepare_inputs_fun,
                  update_inputs_fun,
                  traverse_cache_fun,
                  opts \\ []
                ) do
    {decoder_inputs, decoder_input_ids, max_length} = prepare_inputs_fun.(inputs, params)

    length = Nx.axis_size(decoder_input_ids, 1)

    if length >= max_length do
      raise ArgumentError,
            "the input sequence has #{length} tokens, but max_length is set to #{max_length}." <>
              " Consider increasing :max_new_tokens (or :max_length), or padding the input to a shorter length"
    end

    strategy = opts[:strategy]
    seed = opts[:seed]

    case strategy.type do
      :greedy_search ->
        greedy(
          decoder_inputs,
          decoder_input_ids,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          [max_length: max_length] ++ opts
        )

      :contrastive_search ->
        contrastive(
          decoder_inputs,
          decoder_input_ids,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          traverse_cache_fun,
          [max_length: max_length, top_k: strategy.top_k, penalty_alpha: strategy.alpha] ++ opts
        )

      :multinomial_sampling ->
        prng_key = Nx.Random.key(seed)

        sampling(
          decoder_inputs,
          decoder_input_ids,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          [max_length: max_length, prng_key: prng_key] ++ opts
        )
    end
  end

  # Greedy search

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

    {sequences, length = input_length, finished?} =
      init_sequences(decoder_input_ids, max_length, pad_token_id)

    # The loop works with inputs of length 1, so if the initial input
    # is longer, we make the initial pass outside
    {sequences, length, finished?, inputs} =
      if length > 1 do
        greedy_step(
          sequences,
          length,
          finished?,
          inputs,
          input_length,
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
            continue?(finished?, length, max_length) do
        {sequences, length, finished?, inputs} =
          greedy_step(
            sequences,
            length,
            finished?,
            inputs,
            input_length,
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

  defnp init_sequences(decoder_input_ids, max_length, pad_token_id) do
    {batch_size, length} = Nx.shape(decoder_input_ids)

    if length > max_length do
      raise ArgumentError, "expected the input to be at most #{max_length} tokens, got: #{length}"
    end

    sequences = Nx.broadcast(pad_token_id, {batch_size, max_length})
    sequences = Nx.put_slice(sequences, [0, 0], decoder_input_ids)

    finished? = Nx.broadcast(Nx.tensor(0, type: :u8), {batch_size})

    {sequences, length, finished?}
  end

  defnp continue?(finished?, length, max_length) do
    not Nx.all(finished?) and length < max_length
  end

  defnp greedy_step(
          sequences,
          length,
          finished?,
          inputs,
          input_length,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          opts
        ) do
    pad_token_id = opts[:pad_token_id]
    eos_token_id = opts[:eos_token_id]

    outputs = predict_fun.(params, inputs)

    logits = outputs.logits[[.., -1]]
    logits = batch_process_logits(logits_processor_fun, logits, sequences, length, input_length)
    token_id = Nx.argmax(logits, axis: -1)

    {sequences, length, finished?} =
      update_sequences(sequences, length, finished?, token_id, pad_token_id, eos_token_id)

    inputs = update_inputs_fun.(inputs, outputs.cache, Nx.new_axis(token_id, -1))

    {sequences, length, finished?, inputs}
  end

  defnp update_sequences(sequences, length, finished?, token_id, pad_token_id, eos_token_id) do
    token_id = Nx.select(finished?, pad_token_id, token_id)

    finished? =
      case eos_token_id do
        nil -> finished?
        eos_token_id -> finished? or token_id == eos_token_id
      end

    {token_id, finished?} = hook({token_id, finished?}, :token)

    token_ids = Nx.new_axis(token_id, -1)
    sequences = Nx.put_slice(sequences, [0, length], token_ids)

    {sequences, length + 1, finished?}
  end

  defnp batch_process_logits(logits_processor_fun, logits, sequences, length, input_length) do
    logits
    |> Nx.vectorize(:batch)
    |> logits_processor_fun.(%{
      sequence: Nx.vectorize(sequences, :batch),
      length: length,
      input_length: input_length
    })
    |> Nx.devectorize(keep_names: false)
  end

  # Contrastive search

  defnp contrastive(
          inputs,
          decoder_input_ids,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          traverse_cache_fun,
          opts \\ []
        ) do
    max_length = opts[:max_length]
    pad_token_id = opts[:pad_token_id]
    eos_token_id = opts[:eos_token_id]
    top_k = opts[:top_k]
    penalty_alpha = opts[:penalty_alpha]

    {sequences, length = input_length, finished?} =
      init_sequences(decoder_input_ids, max_length, pad_token_id)

    # Step (1)
    # Initial pass to obtain hidden state and expand inputs to top-k

    outputs = predict_fun.(params, inputs)

    # Later, we feed model a single token at a time and reuse previous
    # results using cache. Here we need the final hidden state, so we
    # need to keep track of it in a similar way
    initial_hidden_state = decoder_hidden_state(outputs)
    batch_size = Nx.axis_size(initial_hidden_state, 0)
    hidden_size = Nx.axis_size(initial_hidden_state, -1)
    joint_hidden_state = Nx.broadcast(0.0, {batch_size, max_length, hidden_size})
    joint_hidden_state = Nx.put_slice(joint_hidden_state, [0, 0, 0], initial_hidden_state)

    logits = outputs.logits[[.., -1]]
    logits = batch_process_logits(logits_processor_fun, logits, sequences, length, input_length)
    scores = Axon.Activations.softmax(logits, axis: -1)
    {top_k_scores, top_k_token_ids} = Nx.top_k(scores, k: top_k)

    # For subsequent model passes we consider several (top-k) paths
    # for each batch item, so we duplicate inputs and cache accordingly
    inputs = expand_inputs(inputs, top_k)
    cache = expand_cache(outputs.cache, top_k, traverse_cache_fun)
    inputs = update_inputs_fun.(inputs, cache, Nx.reshape(top_k_token_ids, {:auto, 1}))

    # Step (2)
    # In the loop we make prediction for top-k continuation tokens and
    # pick the best one using the contrastive rank. From the same model
    # pass we also get the next top-k continuation tokens

    {sequences, _length, _finished?, _inputs, _params, _joint_hidden_state, _top_k_values} =
      while {sequences, length, finished?, inputs, params, joint_hidden_state,
             {top_k_scores, top_k_token_ids}},
            continue?(finished?, length, max_length) do
        outputs = predict_fun.(params, inputs)

        hidden_state = decoder_hidden_state(outputs)

        context_hidden_state = Utils.Nx.repeat_interleave(joint_hidden_state, top_k)

        selected_idx =
          contrastive_rank(
            context_hidden_state,
            hidden_state,
            length,
            top_k_scores,
            penalty_alpha,
            top_k
          )

        hidden_state = Utils.Nx.chunked_take(hidden_state, top_k, selected_idx)
        joint_hidden_state = Nx.put_slice(joint_hidden_state, [0, length, 0], hidden_state)

        token_id = top_k_token_ids |> Nx.flatten() |> Utils.Nx.chunked_take(top_k, selected_idx)

        {sequences, length, finished?} =
          update_sequences(sequences, length, finished?, token_id, pad_token_id, eos_token_id)

        logits = outputs.logits[[.., -1]]
        logits = Utils.Nx.chunked_take(logits, top_k, selected_idx)

        logits =
          batch_process_logits(logits_processor_fun, logits, sequences, length, input_length)

        scores = Axon.Activations.softmax(logits, axis: -1)
        {top_k_scores, top_k_token_ids} = Nx.top_k(scores, k: top_k)

        # Mirror the selected idx to other entries within each chunk
        cache = reflect_cache(outputs.cache, top_k, selected_idx, traverse_cache_fun)
        inputs = update_inputs_fun.(inputs, cache, Nx.reshape(top_k_token_ids, {:auto, 1}))

        {sequences, length, finished?, inputs, params, joint_hidden_state,
         {top_k_scores, top_k_token_ids}}
      end

    sequences
  end

  deftransformp decoder_hidden_state(outputs) do
    hidden_states =
      case outputs do
        %{decoder_hidden_states: hidden_states} -> hidden_states
        %{hidden_states: hidden_states} -> hidden_states
      end

    elem(hidden_states, tuple_size(hidden_states) - 1)
  end

  deftransformp expand_inputs(inputs, times) do
    Map.new(inputs, fn
      {key, value} when key in ["cache"] ->
        {key, value}

      {key, %Nx.Tensor{} = value} ->
        {key, Utils.Nx.repeat_interleave(value, times)}
    end)
  end

  deftransformp expand_cache(cache, times, traverse_cache_fun) do
    traverse_cache_fun.(cache, &Utils.Nx.repeat_interleave(&1, times))
  end

  deftransformp reflect_cache(cache, times, idx, traverse_cache_fun) do
    traverse_cache_fun.(
      cache,
      &(&1
        |> Utils.Nx.chunked_take(times, idx)
        |> Utils.Nx.repeat_interleave(times))
    )
  end

  defnp contrastive_rank(
          context_hidden_state,
          hidden_state,
          length,
          top_k_scores,
          penalty_alpha,
          top_k
        ) do
    similarity_matrix =
      context_hidden_state
      |> Bumblebee.Utils.Nx.cosine_similarity(hidden_state, batched?: true)
      # hidden_state has sequence length of 1, so the batch of similarity
      # matrices has shape {batch_size * top_k, max_length, 1} and we
      # flatten out the last dimension
      |> Nx.squeeze(axes: [-1])

    # context_hidden_state includes placeholder values for tokens up
    # to max_length, so we need to ignore these
    current_sequence? = Nx.iota(Nx.shape(similarity_matrix), axis: -1) < length

    degeneration_penalty =
      current_sequence?
      |> Nx.select(similarity_matrix, Nx.Constants.neg_infinity())
      |> Nx.reduce_max(axes: [-1])

    contrastive_score =
      (1.0 - penalty_alpha) * Nx.flatten(top_k_scores) - penalty_alpha * degeneration_penalty

    contrastive_score
    |> Nx.reshape({:auto, top_k})
    |> Nx.argmax(axis: -1)
  end

  # Multinomial sampling

  defnp sampling(
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
    prng_key = opts[:prng_key]

    {sequences, length = input_length, finished?} =
      init_sequences(decoder_input_ids, max_length, pad_token_id)

    # The loop works with inputs of length 1, so if the initial input
    # is longer, we make the initial pass outside
    {sequences, length, finished?, inputs, prng_key} =
      if length > 1 do
        sampling_step(
          sequences,
          length,
          finished?,
          inputs,
          input_length,
          predict_fun,
          params,
          prng_key,
          logits_processor_fun,
          update_inputs_fun,
          pad_token_id: pad_token_id,
          eos_token_id: eos_token_id
        )
      else
        {sequences, length, finished?, inputs, prng_key}
      end

    {sequences, _length, _finished?, _inputs, _params, _key} =
      while {sequences, length, finished?, inputs, params, prng_key},
            continue?(finished?, length, max_length) do
        {sequences, length, finished?, inputs, prng_key} =
          sampling_step(
            sequences,
            length,
            finished?,
            inputs,
            input_length,
            predict_fun,
            params,
            prng_key,
            logits_processor_fun,
            update_inputs_fun,
            pad_token_id: pad_token_id,
            eos_token_id: eos_token_id
          )

        {sequences, length, finished?, inputs, params, prng_key}
      end

    sequences
  end

  defnp sampling_step(
          sequences,
          length,
          finished?,
          inputs,
          input_length,
          predict_fun,
          params,
          prng_key,
          logits_processor_fun,
          update_inputs_fun,
          opts \\ []
        ) do
    pad_token_id = opts[:pad_token_id]
    eos_token_id = opts[:eos_token_id]

    key = Nx.Random.split(prng_key)
    {key, prng_key} = {key[1], key[0]}

    outputs = predict_fun.(params, inputs)

    logits = outputs.logits[[.., -1]]
    logits = batch_process_logits(logits_processor_fun, logits, sequences, length, input_length)
    scores = Axon.Activations.softmax(logits)
    token_id = batched_choice(key, scores)

    {sequences, length, finished?} =
      update_sequences(sequences, length, finished?, token_id, pad_token_id, eos_token_id)

    inputs = update_inputs_fun.(inputs, outputs.cache, Nx.new_axis(token_id, -1))

    {sequences, length, finished?, inputs, prng_key}
  end

  deftransformp batched_choice(key, scores) do
    {batch_size, vocab_size} = Nx.shape(scores)

    vocab = Nx.iota({vocab_size})

    keys = Nx.Random.split(key, parts: batch_size)

    key = Nx.vectorize(keys, :batch)
    probabilities = Nx.vectorize(scores, :batch)

    {tokens, _} = Nx.Random.choice(key, vocab, probabilities, samples: 1)

    tokens
    |> Nx.squeeze()
    |> Nx.devectorize()
  end

  # Serving

  @doc false
  def generation(model_info, tokenizer, %Text.GenerationConfig{} = generation_config, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :seed,
        :compile,
        defn_options: [],
        preallocate_params: false,
        stream: false
      ])

    %{model: model, params: params, spec: spec} = model_info

    Shared.validate_architecture!(spec, [
      :for_conditional_generation,
      :for_causal_language_modeling
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

    generate_fun = build_generate(model, spec, generation_config, Keyword.take(opts, [:seed]))

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          generate_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      if opts[:stream] do
        Shared.validate_input_for_stream!(input)
      end

      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts,
            length: sequence_length,
            pad_direction: :left,
            return_token_type_ids: false
          )
        end)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, multi?}
    end)
    |> maybe_stream(opts[:stream], tokenizer)
  end

  defp maybe_stream(serving, false, tokenizer) do
    Nx.Serving.client_postprocessing(serving, fn {token_ids, _metadata}, multi? ->
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      decoded
      |> Enum.map(&%{results: [%{text: &1}]})
      |> Shared.normalize_output(multi?)
    end)
  end

  defp maybe_stream(serving, true, tokenizer) do
    serving
    |> Nx.Serving.streaming(hooks: [:token])
    |> Nx.Serving.client_postprocessing(fn stream, false = _multi? ->
      Stream.transform(stream, %{tokens: [], consumed_size: 0, finished?: false}, fn
        _event, %{finished?: true} = state ->
          {:halt, state}

        {:token, {token_id, finished?}}, state ->
          token_id = Nx.to_number(token_id[0])
          finished? = Nx.to_number(finished?[0]) == 1

          state = %{state | tokens: state.tokens ++ [token_id], finished?: finished?}

          chunk = pending_chunk(tokenizer, state)

          cond do
            # When the sequence is finished early or we reach a newline,
            # we flush the cache
            finished? or String.ends_with?(chunk, "\n") ->
              {[chunk], %{state | tokens: [], consumed_size: 0}}

            # CJK characters are tokenized atomically, so we can emit
            # the chunk
            chunk != "" and cjk_codepoint?(last_codepoint(chunk)) ->
              state = update_in(state.consumed_size, &(&1 + byte_size(chunk)))
              {[chunk], state}

            # Emit chunk until the space. We need to keep tokens,
            # because certain tokenizers do not encode whitespace in
            # tokens and they add a space based on previous tokens
            space_idx = find_last_occurrence(chunk, " ") ->
              if space_idx > 0 do
                chunk = binary_slice(chunk, 0, space_idx)
                state = update_in(state.consumed_size, &(&1 + space_idx))
                {[chunk], state}
              else
                {[], state}
              end

            true ->
              {[], state}
          end

        {:batch, _, _}, state ->
          chunk = pending_chunk(tokenizer, state)

          if chunk == "" do
            {:halt, state}
          else
            {[chunk], %{state | tokens: [], consumed_size: 0}}
          end
      end)
    end)
  end

  defp pending_chunk(tokenizer, state) do
    text = Bumblebee.Tokenizer.decode(tokenizer, state.tokens)
    binary_slice(text, state.consumed_size..-1//1)
  end

  defp find_last_occurrence(string, pattern) do
    case :binary.matches(string, pattern) do
      [] -> nil
      matches -> matches |> List.last() |> elem(0)
    end
  end

  defp last_codepoint(<<codepoint::utf8>>), do: codepoint
  defp last_codepoint(<<_::utf8, rest::binary>>), do: last_codepoint(rest)

  defp cjk_codepoint?(codepoint) do
    # The specific ranges originated in [1] and are generally mirrored
    # in other tokenizers using WordPiece. Also see [2].
    #
    # [1]: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L264-L284
    # [2]: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/multilingual.md#tokenization

    codepoint in 0x4E00..0x9FFF or
      codepoint in 0x3400..0x4DBF or
      codepoint in 0x20000..0x2A6DF or
      codepoint in 0x2A700..0x2B73F or
      codepoint in 0x2B740..0x2B81F or
      codepoint in 0x2B820..0x2CEAF or
      codepoint in 0xF900..0xFAFF or
      codepoint in 0x2F800..0x2FA1F
  end
end
