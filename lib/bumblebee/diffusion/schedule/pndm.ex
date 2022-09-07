defmodule Bumblebee.Diffusion.Schedule.Pndm do
  @derive {Nx.Container,
           containers: [:state],
           keep: [
             :num_inference_steps,
             :num_train_timesteps,
             :offset,
             :beta_start,
             :beta_end,
             :beta_schedule,
             :skip_prk_steps
           ]}
  defstruct num_inference_steps: nil,
            num_train_timesteps: 1000,
            offset: 0,
            beta_start: 0.0001,
            beta_end: 0.02,
            # TODO: support other variants
            beta_schedule: :linear,
            # TODO: work this in
            skip_prk_steps: false,
            state: nil

  import Nx.Defn

  def new(num_inference_steps, latents_shape, opts \\ []) do
    state = %{
      ets:
        {Nx.broadcast(0.0, latents_shape), Nx.broadcast(0.0, latents_shape),
         Nx.broadcast(0.0, latents_shape), Nx.broadcast(0.0, latents_shape)},
      ets_size: 0,
      counter: 0,
      cur_sample: Nx.broadcast(0.0, latents_shape)
    }

    scheduler = %__MODULE__{
      num_inference_steps: num_inference_steps,
      state: state
    }

    Bumblebee.Shared.put_config_attrs(scheduler, opts)
  end

  def timesteps(scheduler) do
    step = div(scheduler.num_train_timesteps, scheduler.num_inference_steps)

    0..(scheduler.num_inference_steps - 1)
    |> Enum.map(fn i -> i * step + scheduler.offset end)
    |> Enum.reverse()
    |> then(fn [t0, t1 | ts] -> [t0, t1, t1 | ts] end)
  end

  defn step(scheduler, model_output, timestep, sample) do
    state = scheduler.state

    num_inference_steps = scheduler.num_inference_steps

    beta_start = scheduler.beta_start
    beta_end = scheduler.beta_end
    num_train_timesteps = scheduler.num_train_timesteps

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

    {%{scheduler | state: state}, prev_sample}
  end

  deftransform add_ets(state, model_output) do
    ets_size = Nx.min(tuple_size(state.ets), Nx.add(state.ets_size, 1))

    ets =
      state.ets
      |> Tuple.delete_at(tuple_size(state.ets) - 1)
      |> Tuple.insert_at(0, model_output)

    %{state | ets: ets, ets_size: ets_size}
  end

  defnp linspace(start, stop, opts \\ []) do
    opts = keyword!(opts, [:steps])
    n = opts[:steps]

    step_size = (stop - start) / (n - 1)
    Nx.iota({n}) * step_size + start
  end
end
