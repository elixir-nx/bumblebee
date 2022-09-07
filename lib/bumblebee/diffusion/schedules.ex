defmodule Bumblebee.Diffusion.Schedules do
  import Nx.Defn

  # TODO: It migth be better to take a module appraoch
  # per scheduler

  def pndm(num_inference_steps, latents_shape, opts \\ []) do
    state = %{
      ets:
        {Nx.broadcast(0.0, latents_shape), Nx.broadcast(0.0, latents_shape),
         Nx.broadcast(0.0, latents_shape), Nx.broadcast(0.0, latents_shape)},
      ets_size: 0,
      counter: 0,
      cur_sample: Nx.broadcast(0.0, latents_shape)
    }

    # TODO: Create schedule from config
    # TODO: state to struct, use protocols for schedulers
    {state, &step_pndm(&1, &2, &3, &4, [{:num_inference_steps, num_inference_steps} | opts])}
  end

  defnp step_pndm(model_output, timestep, sample, state, opts \\ []) do
    opts =
      keyword!(opts, [
        :num_inference_steps,
        num_train_timesteps: 1000,
        beta_start: 0.0001,
        beta_end: 0.02,
        # TODO: support other variants
        beta_schedule: :linear,
        # TODO: work this in
        skip_prk_steps: false
      ])

    num_inference_steps = opts[:num_inference_steps]

    beta_start = opts[:beta_start]
    beta_end = opts[:beta_end]
    num_train_timesteps = opts[:num_train_timesteps]

    # TODO: other variants, this is :scaled_linear
    betas = linspace(beta_start ** 0.5, beta_end ** 0.5, steps: num_train_timesteps) ** 2
    alphas = 1 - betas
    alphas_cumprod = Nx.cumulative_product(alphas, axis: 0)

    offset = 1

    # TODO: configurable offset
    # TODO: in diffusers they don't always have num_inference_steps of timestamps,
    # which version is correct?
    # timesteps =
    #   Nx.iota({num_inference_steps})
    #   |> Nx.multiply(Nx.quotient(1000, num_inference_steps))
    #   |> Nx.reverse()
    #   |> Nx.add(offset)

    # TODO: currently we always do skip_prk_steps: true, so below goes step_plms
    prev_timestep = Nx.max(timestep - Nx.quotient(num_train_timesteps, num_inference_steps), 0)

    {state, prev_timestep, timestep} =
      if state.counter != 1 do
        {add_ets(state, model_output), prev_timestep, timestep}
      else
        {state, timestep, timestep + Nx.quotient(num_train_timesteps, num_inference_steps)}
      end

    {state, sample, model_output} =
      cond do
        state.ets_size == 1 and state.counter == 0 ->
          {%{state | cur_sample: sample}, sample, model_output}

        state.ets_size == 1 and state.counter == 1 ->
          model_output = (model_output + elem(state.ets, 0)) / 2
          {state, state.cur_sample, model_output}

        state.ets_size == 2 ->
          model_output = (3 * elem(state.ets, 0) - elem(state.ets, 1)) / 2
          {state, sample, model_output}

        state.ets_size == 3 ->
          model_output =
            (23 * elem(state.ets, 0) - 16 * elem(state.ets, 1) + 5 * elem(state.ets, 2)) / 12

          {state, sample, model_output}

        true ->
          model_output =
            (55 * elem(state.ets, 0) - 59 * elem(state.ets, 1) + 37 * elem(state.ets, 2) -
               9 * elem(state.ets, 3)) / 24

          {state, sample, model_output}
      end

    alpha_prod_t = alphas_cumprod[timestep + 1 - offset]
    alpha_prod_t_prev = alphas_cumprod[prev_timestep + 1 - offset]
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** 0.5

    model_output_denom_coeff =
      alpha_prod_t * beta_prod_t_prev ** 0.5 +
        (alpha_prod_t * beta_prod_t * alpha_prod_t_prev) ** 0.5

    prev_sample =
      sample_coeff * sample -
        (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff

    state = %{state | counter: state.counter + 1}

    {state, prev_sample}
  end

  deftransform add_ets(state, model_output) do
    ets_size = Nx.min(tuple_size(state.ets), Nx.add(state.ets_size, 1))

    ets =
      state.ets
      |> Tuple.delete_at(tuple_size(state.ets) - 1)
      |> Tuple.insert_at(0, model_output)

    %{state | ets: ets, ets_size: ets_size}
  end

  def ddim(num_inference_steps) do
    # TODO: Create schedule from config
    &step_ddim(&1, &2, &3, num_inference_steps: num_inference_steps)
  end

  defnp step_ddim(model_output, timestep, sample, opts \\ []) do
    opts =
      keyword!(opts, [
        :num_inference_steps,
        num_train_timesteps: 1000,
        beta_start: 0.0001,
        beta_end: 0.02,
        beta_schedule: :linear,
        clip_sample: true,
        use_clipped_model_output: false,
        eta: 0.0
      ])

    beta_start = opts[:beta_start]
    beta_end = opts[:beta_end]
    beta_schedule = opts[:beta_schedule]
    num_train_timesteps = opts[:num_train_timesteps]
    num_inference_steps = opts[:num_inference_steps]
    clip_sample = opts[:clip_sample]
    use_clipped_model_output = opts[:use_clipped_model_output]
    eta = opts[:eta]

    betas = linspace(beta_start, beta_end, steps: num_train_timesteps)
    alphas = 1 - betas
    alphas_cumprod = Nx.cumulative_product(alphas, axis: 0)

    prev_time_step = timestep - Nx.quotient(num_train_timesteps, num_inference_steps)
    alpha_prod_t = alphas_cumprod[[timestep]]

    alpha_prod_t_prev =
      if prev_time_step >= 0 do
        alphas_cumprod[[prev_time_step]]
      else
        Nx.tensor(1.0)
      end

    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - Nx.sqrt(beta_prod_t) * model_output) / Nx.sqrt(alpha_prod_t)

    pred_original_sample =
      if clip_sample do
        Nx.clip(pred_original_sample, -1, 1)
      else
        pred_original_sample
      end

    variance = get_variance(timestep, prev_time_step)
    std_dev_t = eta * Nx.sqrt(variance)

    model_output =
      if use_clipped_model_output do
        (sample - Nx.sqrt(alpha_prod_t) * pred_original_sample) / Nx.sqrt(beta_prod_t)
      else
        model_output
      end

    pred_sample_direction = Nx.sqrt(1 - alpha_prod_t_prev - Nx.power(std_dev_t, 2)) * model_output

    prev_sample = Nx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

    if eta > 0 do
      noise = Nx.random_normal(Nx.shape(model_output))
      variance = Nx.sqrt(get_variance(timestep, prev_time_step)) * eta * noise
      prev_sample + variance
    else
      prev_sample
    end
  end

  ## Helpers

  defnp linspace(start, stop, opts \\ []) do
    opts = keyword!(opts, [:steps])
    n = opts[:steps]

    step_size = (stop - start) / (n - 1)
    Nx.iota({n}) * step_size + start
  end

  defnp get_variance(timestep, prev_time_step, opts \\ []) do
    opts =
      keyword!(opts,
        num_train_timesteps: 1000,
        beta_start: 0.0001,
        beta_end: 0.02,
        beta_schedule: :linear
      )

    beta_start = opts[:beta_start]
    beta_end = opts[:beta_end]
    num_train_timesteps = opts[:num_train_timesteps]

    betas = linspace(beta_start, beta_end, steps: num_train_timesteps)
    alphas = 1 - betas
    alphas_cumprod = Nx.cumulative_product(alphas, axis: 0)

    alpha_prod_t = alphas_cumprod[[timestep]]

    alpha_prod_t_prev =
      if prev_time_step >= 0 do
        alphas_cumprod[[prev_time_step]]
      else
        Nx.tensor(1.0)
      end

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t / alpha_prod_t_prev)
  end
end
