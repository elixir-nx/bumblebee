defmodule Bumblebee.Diffusion.PndmScheduler do
  @moduledoc ~S"""
  Pseudo numerical methods for diffusion models (PNDMs).

  The sampling is based on two numerical methods for solving ODE: the
  Runge-Kutta method (RK) and the linear multi-step method (LMS). The
  gradient at each step is computed according to either of these methods,
  however the transfer part (approximating the next sample based on
  current sample and gradient) is non-linear. Because of this property,
  the authors of the paper refer to them as pseudo numerical methods,
  denoted as PRK and PLMS respectively.

  ## Configuration

    * `:num_train_timesteps` - the number of diffusion steps used to
      train the model. Defaults to `1000`

    * `:beta_schedule` - the beta schedule type, a mapping from a beta
      range to a sequence of betas for stepping the model. Either of
      `:linear`, `:scaled_linear`, or `:squaredcos_cap_v2`. Defaults to
      `:linear`

    * `:beta_start` - the start value for the beta schedule. Defaults
      to `0.0001`

    * `:beta_end` - the end value for the beta schedule. Defaults to `0.02`

    * `:set_alpha_to_one` - each step $t$ uses the values of $\bar{\alpha}\_t$
      and $\bar{\alpha}\_{t-1}$, however for $t = 0$ there is no previous
      alpha. Setting this option to `true` implies $\bar{\alpha}\_{t-1} = 1$,
      otherwise $\bar{\alpha}\_{t-1} = \bar{\alpha}\_0$. Defaults to `false`

    * `:steps_offset` - an offset added to the inference steps. You can
      use a combination of `offset: 1` and `set_alpha_to_one: false`,
      so that the last step $t = 1$ uses $\bar{\alpha}\_1$ and $\bar{\alpha}\_0$,
      as done in stable diffusion. Defaults to `0`

    * `:skip_prk_steps` - when `true`, the first few samples are computed
      using lower-order linear multi-step, rather than the Runge-Kutta
      method. This results in less forward passes of the model. Defaults
      to `false`

  ## References

    * [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/pdf/2202.09778.pdf)

  """

  import Nx.Defn

  alias Bumblebee.Diffusion.SchedulerUtils

  @behaviour Bumblebee.Scheduler

  defstruct num_train_timesteps: 1000,
            beta_schedule: :linear,
            beta_start: 0.0001,
            beta_end: 0.02,
            set_alpha_to_one: false,
            steps_offset: 0,
            skip_prk_steps: false

  @impl true
  def config(config, opts) do
    Bumblebee.Shared.put_config_attrs(config, opts)
  end

  @impl true
  def init(config, num_inference_timesteps, sample_shape) do
    timesteps =
      timesteps(
        config.num_train_timesteps,
        num_inference_timesteps,
        config.steps_offset,
        config.skip_prk_steps
      )

    alpha_bars = init_parameters(config: config)

    empty = Nx.broadcast(0.0, sample_shape)

    state = %{
      timesteps: timesteps,
      timestep_gap: div(config.num_train_timesteps, num_inference_timesteps),
      alpha_bars: alpha_bars,
      iteration: 0,
      recent_noise: empty |> List.duplicate(4) |> List.to_tuple(),
      current_sample: empty,
      noise_prime: empty
    }

    {state, timesteps}
  end

  defnp init_parameters(opts \\ []) do
    %{
      beta_start: beta_start,
      beta_end: beta_end,
      beta_schedule: beta_schedule,
      num_train_timesteps: num_train_timesteps
    } = opts[:config]

    betas =
      SchedulerUtils.beta_schedule(beta_schedule, num_train_timesteps,
        linear_start: beta_start,
        linear_end: beta_end
      )

    alphas = 1 - betas

    Nx.cumulative_product(alphas)
  end

  deftransformp timesteps(num_train_timesteps, num_inference_timesteps, offset, skip_prk_steps) do
    # Note that there are more timesteps than `num_inference_timesteps`.
    # That's because each timestep corresponds to a single forward pass
    # of the denoising model and the first few steps require multiple such
    # passes. In other words, the timesteps list is used for subsequent
    # model calls, while the actual sampling happens at the same timesteps
    # as with DDIM.

    timestep_gap = div(num_train_timesteps, num_inference_timesteps)

    ddim_timesteps =
      SchedulerUtils.ddim_timesteps(num_train_timesteps, num_inference_timesteps, offset)

    if skip_prk_steps do
      just_plms_timesteps(ddim_timesteps, timestep_gap)
    else
      prk_plms_timesteps(ddim_timesteps, timestep_gap)
    end
  end

  defnp prk_plms_timesteps(ddim_timesteps, timestep_gap) do
    if Nx.size(ddim_timesteps) < 4 do
      prk_timesteps(ddim_timesteps, timestep_gap)
    else
      prk_timesteps = prk_timesteps(ddim_timesteps[0..2//1], timestep_gap)
      plms_timesteps = ddim_timesteps[3..-1//1]
      Nx.concatenate([prk_timesteps, plms_timesteps])
    end
  end

  defnp prk_timesteps(timesteps, timestep_gap) do
    deltas = Nx.stack([0, div(timestep_gap, 2), div(timestep_gap, 2), timestep_gap])

    timesteps
    |> Nx.reshape({:auto, 1})
    |> Nx.subtract(Nx.reshape(deltas, {1, :auto}))
    |> Nx.flatten()
  end

  defnp just_plms_timesteps(ddim_timesteps, timestep_gap) do
    leading_timesteps = Nx.stack([ddim_timesteps[0], ddim_timesteps[0] - timestep_gap])

    if Nx.size(ddim_timesteps) < 2 do
      leading_timesteps
    else
      Nx.concatenate([leading_timesteps, ddim_timesteps[1..-1//1]])
    end
  end

  @impl true
  deftransform step(config, state, sample, noise) do
    do_step(config, state, sample, noise)
  end

  defnp do_step(config \\ [], state, sample, noise) do
    {state, prev} =
      if config.skip_prk_steps do
        step_just_plms(config, state, sample, noise)
      else
        step_prk_plms(config, state, sample, noise)
      end

    state = %{state | iteration: state.iteration + 1}

    {state, prev}
  end

  defnp step_prk_plms(config, state, sample, noise) do
    # This is the version from the original paper [1], specifically F-PNDM.
    # It uses the Runge-Kutta method to compute the first 3 results (each
    # requiring 4 iterations).
    #
    # [1]: https://arxiv.org/abs/2202.09778

    if state.iteration < 12 do
      step_prk(config, state, sample, noise)
    else
      step_plms(config, state, sample, noise)
    end
  end

  defnp step_just_plms(config, state, sample, noise) do
    # This alternative version is based on the paper, however instead of the
    # Runge-Kutta method, it uses lower-order linear multi-step for computing
    # the first 3 results (2, 1, 1 iterations respectively). For the original
    # implementation see [1].
    #
    # [1]: https://github.com/CompVis/latent-diffusion/pull/51

    if state.iteration < 4 do
      step_warmup_plms(config, state, sample, noise)
    else
      step_plms(config, state, sample, noise)
    end
  end

  # # Note on notation
  #
  # The paper denotes sample as x_t, noise as e_t, model as eps, prev_sample
  # function as phi. The superscript in case of x_t and e_t translates to
  # consecutive iterations, since we have one iteration per model forward
  # pass (the eps function). We keep track of x_t as current_sample, and
  # noise_prime corresponds to e_t prime.

  defnp step_prk(config, state, sample, noise) do
    # See Equation (13)

    %{noise_prime: noise_prime, current_sample: current_sample} = state

    rk_step_number = rem(state.iteration, 4)

    state =
      if rk_step_number == 0 do
        store_noise(state, noise)
      else
        state
      end

    {noise_prime, current_sample, noise} =
      cond do
        rk_step_number == 0 ->
          noise_prime = noise_prime + noise / 6
          {noise_prime, sample, noise}

        rk_step_number == 1 ->
          noise_prime = noise_prime + noise / 3
          {noise_prime, current_sample, noise}

        rk_step_number == 2 ->
          noise_prime = noise_prime + noise / 3
          {noise_prime, current_sample, noise}

        true ->
          noise_prime = noise_prime + noise / 6
          {0, current_sample, noise_prime}
      end

    state = %{state | current_sample: current_sample, noise_prime: noise_prime}

    timestep = state.timesteps[state.iteration - rk_step_number]
    diff = if(rk_step_number < 2, do: div(state.timestep_gap, 2), else: state.timestep_gap)
    prev_timestep = timestep - diff

    prev_sample = prev_sample(config, state, current_sample, noise, timestep, prev_timestep)

    {state, prev_sample}
  end

  defnp step_warmup_plms(config, state, sample, noise) do
    # The first two iterations use Equation (22), third iteration uses
    # Equation (23), and fourth iteration uses third-order LMS in the
    # same spirit.

    %{current_sample: current_sample} = state

    state =
      if state.iteration != 1 do
        store_noise(state, noise)
      else
        state
      end

    {current_sample, noise} =
      cond do
        state.iteration == 0 ->
          {sample, noise}

        state.iteration == 1 ->
          noise_prime = (noise + elem(state.recent_noise, 0)) / 2
          {current_sample, noise_prime}

        state.iteration == 2 ->
          noise_prime = (3 * elem(state.recent_noise, 0) - elem(state.recent_noise, 1)) / 2
          {sample, noise_prime}

        true ->
          noise_prime =
            (23 * elem(state.recent_noise, 0) - 16 * elem(state.recent_noise, 1) +
               5 * elem(state.recent_noise, 2)) / 12

          {sample, noise_prime}
      end

    state = %{state | current_sample: current_sample}

    timestep =
      if state.iteration == 1 do
        state.timesteps[state.iteration - 1]
      else
        state.timesteps[state.iteration]
      end

    prev_timestep = timestep - state.timestep_gap

    prev_sample = prev_sample(config, state, current_sample, noise, timestep, prev_timestep)

    {state, prev_sample}
  end

  defnp step_plms(config, state, sample, noise) do
    # See Equation (12)

    state = store_noise(state, noise)

    noise =
      (55 * elem(state.recent_noise, 0) - 59 * elem(state.recent_noise, 1) +
         37 * elem(state.recent_noise, 2) - 9 * elem(state.recent_noise, 3)) / 24

    timestep = state.timesteps[state.iteration]
    prev_timestep = timestep - state.timestep_gap

    prev_sample = prev_sample(config, state, sample, noise, timestep, prev_timestep)

    {state, prev_sample}
  end

  defnp prev_sample(config, state, sample, noise, timestep, prev_timestep) do
    # See Equation (11)

    alpha_bar_t = state.alpha_bars[timestep]

    alpha_bar_t_prev =
      if prev_timestep >= 0 do
        state.alpha_bars[prev_timestep]
      else
        if config.set_alpha_to_one, do: 1.0, else: state.alpha_bars[0]
      end

    sample_coeff = (alpha_bar_t_prev / alpha_bar_t) ** 0.5

    noise_coeff = alpha_bar_t_prev - alpha_bar_t

    noise_denom_coeff =
      alpha_bar_t * (1 - alpha_bar_t_prev) ** 0.5 +
        (alpha_bar_t * (1 - alpha_bar_t) * alpha_bar_t_prev) ** 0.5

    sample_coeff * sample - noise_coeff * noise / noise_denom_coeff
  end

  deftransformp store_noise(state, noise) do
    recent_noise =
      state.recent_noise
      |> Tuple.delete_at(tuple_size(state.recent_noise) - 1)
      |> Tuple.insert_at(0, noise)

    %{state | recent_noise: recent_noise}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    alias Bumblebee.Shared

    def load(config, data) do
      data
      |> Shared.convert_to_atom(["beta_schedule"])
      |> Shared.data_into_config(config)
    end
  end
end
