defmodule Bumblebee.Diffusion.PndmScheduler do
  options = [
    num_train_steps: [
      default: 1000,
      doc: "the number of diffusion steps used to train the model"
    ],
    beta_schedule: [
      default: :linear,
      doc: """
      the beta schedule type, a mapping from a beta range to a sequence of betas for stepping the model.
      Either of `:linear`, `:quadratic`, or `:squared_cosine`
      """
    ],
    beta_start: [
      default: 0.0001,
      doc: "the start value for the beta schedule"
    ],
    beta_end: [
      default: 0.02,
      doc: "the end value for the beta schedule"
    ],
    prediction_type: [
      default: :noise,
      doc: """
      prediction type of the denoising model. Either of:

        * `:noise` (default) - the model predicts the noise of the diffusion process

        * `:angular_velocity` - the model predicts velocity in angular parameterization.
          See Section 2.4 in [Imagen Video: High Definition Video Generation with Diffusion Models](https://imagen.research.google/video/paper.pdf),
          then Section 4 in [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/pdf/2202.00512.pdf)
          and Appendix D

      """
    ],
    alpha_clip_strategy: [
      default: :alpha_zero,
      doc: ~S"""
      each step $t$ uses the values of $\bar{\alpha}\_t$ and $\bar{\alpha}\_{t-1}$,
      however for $t = 0$ there is no previous alpha. The strategy can be either
      `:one` ($\bar{\alpha}\_{t-1} = 1$) or `:alpha_zero` ($\bar{\alpha}\_{t-1} = \bar{\alpha}\_0$)
      """
    ],
    timesteps_offset: [
      default: 0,
      doc: ~S"""
      an offset added to the inference steps. You can use a combination of `timesteps_offset: 1` and
      `alpha_clip_strategy: :alpha_zero`, so that the last step $t = 1$ uses $\bar{\alpha}\_1$
      and $\bar{\alpha}\_0$, as done in stable diffusion
      """
    ],
    reduce_warmup: [
      default: false,
      doc: """
      when `true`, the first few samples are computed using lower-order linear multi-step,
      rather than the Runge-Kutta method. This results in less forward passes of the model
      """
    ]
  ]

  @moduledoc """
  Pseudo numerical methods for diffusion models (PNDMs).

  The sampling is based on two numerical methods for solving ODE: the
  Runge-Kutta method (RK) and the linear multi-step method (LMS). The
  gradient at each step is computed according to either of these methods,
  however the transfer part (approximating the next sample based on
  current sample and gradient) is non-linear. Because of this property,
  the authors of the paper refer to them as pseudo numerical methods,
  denoted as PRK and PLMS respectively.

  ## Configuration

  #{Bumblebee.Shared.options_doc(options)}

  ## References

    * [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778)

  """

  defstruct Bumblebee.Shared.option_defaults(options)

  @behaviour Bumblebee.Scheduler
  @behaviour Bumblebee.Configurable

  import Nx.Defn

  alias Bumblebee.Diffusion.SchedulerUtils

  @impl true
  def config(scheduler, opts) do
    Bumblebee.Shared.put_config_attrs(scheduler, opts)
  end

  @impl true
  def init(scheduler, num_steps, sample_template, _prng_key) do
    timesteps =
      timesteps(
        scheduler.num_train_steps,
        num_steps,
        scheduler.timesteps_offset,
        scheduler.reduce_warmup
      )

    alpha_bars = init_parameters(scheduler: scheduler)

    empty = Nx.fill(sample_template, 0)

    state = %{
      timesteps: timesteps,
      timestep_gap: div(scheduler.num_train_steps, num_steps),
      alpha_bars: alpha_bars,
      iteration: 0,
      recent_prediction: empty |> List.duplicate(4) |> List.to_tuple(),
      current_sample: empty,
      prediction_prime: empty
    }

    {state, timesteps}
  end

  defnp init_parameters(opts \\ []) do
    %{
      beta_start: beta_start,
      beta_end: beta_end,
      beta_schedule: beta_schedule,
      num_train_steps: num_train_steps
    } = opts[:scheduler]

    betas =
      SchedulerUtils.beta_schedule(beta_schedule, num_train_steps,
        start: beta_start,
        end: beta_end
      )

    alphas = 1 - betas

    Nx.cumulative_product(alphas)
  end

  deftransformp timesteps(num_train_steps, num_steps, offset, reduce_warmup) do
    # Note that there are more timesteps than `num_steps`.
    # That's because each timestep corresponds to a single forward pass
    # of the denoising model and the first few steps require multiple such
    # passes. In other words, the timesteps list is used for subsequent
    # model calls, while the actual sampling happens at the same timesteps
    # as with DDIM.

    timestep_gap = div(num_train_steps, num_steps)

    ddim_timesteps = SchedulerUtils.ddim_timesteps(num_train_steps, num_steps, offset)

    if reduce_warmup do
      if num_steps < 2 do
        raise ArgumentError,
              "expected at least 2 steps when using :reduce_warmup, got: #{inspect(num_steps)}"
      end

      just_plms_timesteps(ddim_timesteps, timestep_gap)
    else
      if num_steps < 4 do
        raise ArgumentError, "expected at least 4 steps, got: #{inspect(num_steps)}"
      end

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
  def step(scheduler, state, sample, prediction) do
    do_step(state, sample, prediction, scheduler: scheduler)
  end

  defnp do_step(state, sample, prediction, opts) do
    scheduler = opts[:scheduler]

    {state, prev} =
      if scheduler.reduce_warmup do
        step_just_plms(scheduler, state, sample, prediction)
      else
        step_prk_plms(scheduler, state, sample, prediction)
      end

    state = %{state | iteration: state.iteration + 1}

    {state, prev}
  end

  defnp step_prk_plms(scheduler, state, sample, prediction) do
    # This is the version from the original paper [1], specifically F-PNDM.
    # It uses the Runge-Kutta method to compute the first 3 results (each
    # requiring 4 iterations).
    #
    # [1]: https://arxiv.org/abs/2202.09778

    if state.iteration < 12 do
      step_prk(scheduler, state, sample, prediction)
    else
      step_plms(scheduler, state, sample, prediction)
    end
  end

  defnp step_just_plms(scheduler, state, sample, prediction) do
    # This alternative version is based on the paper, however instead of the
    # Runge-Kutta method, it uses lower-order linear multi-step for computing
    # the first 3 results (2, 1, 1 iterations respectively). For the original
    # implementation see [1].
    #
    # [1]: https://github.com/CompVis/latent-diffusion/pull/51

    if state.iteration < 4 do
      step_warmup_plms(scheduler, state, sample, prediction)
    else
      step_plms(scheduler, state, sample, prediction)
    end
  end

  # # Note on notation
  #
  # The paper denotes sample as x_t, prediction as e_t, model as eps,
  # prev_sample function as phi. The superscript in case of x_t and e_t
  # translates to consecutive iterations, since we have one iteration
  # per model forward pass (the eps function). We keep track of x_t as
  # current_sample, and prediction_prime corresponds to e_t prime.

  defnp step_prk(scheduler, state, sample, prediction) do
    # See Equation (13)

    %{prediction_prime: prediction_prime, current_sample: current_sample} = state

    rk_step_number = rem(state.iteration, 4)

    state =
      if rk_step_number == 0 do
        store_prediction(state, prediction)
      else
        state
      end

    {prediction_prime, current_sample, prediction} =
      cond do
        rk_step_number == 0 ->
          prediction_prime = prediction_prime + prediction / 6
          {prediction_prime, sample, prediction}

        rk_step_number == 1 ->
          prediction_prime = prediction_prime + prediction / 3
          {prediction_prime, current_sample, prediction}

        rk_step_number == 2 ->
          prediction_prime = prediction_prime + prediction / 3
          {prediction_prime, current_sample, prediction}

        true ->
          prediction_prime = prediction_prime + prediction / 6
          {Nx.broadcast(0.0, prediction_prime), current_sample, prediction_prime}
      end

    state = %{state | current_sample: current_sample, prediction_prime: prediction_prime}

    timestep = state.timesteps[state.iteration - rk_step_number]
    diff = if(rk_step_number < 2, do: div(state.timestep_gap, 2), else: state.timestep_gap)
    prev_timestep = timestep - diff

    prev_sample =
      prev_sample(scheduler, state, current_sample, prediction, timestep, prev_timestep)

    {state, prev_sample}
  end

  defnp step_warmup_plms(scheduler, state, sample, prediction) do
    # The first two iterations use Equation (22), third iteration uses
    # Equation (23), and fourth iteration uses third-order LMS in the
    # same spirit.

    %{current_sample: current_sample} = state

    state =
      if state.iteration != 1 do
        store_prediction(state, prediction)
      else
        state
      end

    {current_sample, prediction} =
      cond do
        state.iteration == 0 ->
          {sample, prediction}

        state.iteration == 1 ->
          prediction_prime = (prediction + elem(state.recent_prediction, 0)) / 2
          {current_sample, prediction_prime}

        state.iteration == 2 ->
          prediction_prime =
            (3 * elem(state.recent_prediction, 0) - elem(state.recent_prediction, 1)) / 2

          {sample, prediction_prime}

        true ->
          prediction_prime =
            (23 * elem(state.recent_prediction, 0) - 16 * elem(state.recent_prediction, 1) +
               5 * elem(state.recent_prediction, 2)) / 12

          {sample, prediction_prime}
      end

    state = %{state | current_sample: current_sample}

    timestep =
      if state.iteration == 1 do
        state.timesteps[state.iteration - 1]
      else
        state.timesteps[state.iteration]
      end

    prev_timestep = timestep - state.timestep_gap

    prev_sample =
      prev_sample(scheduler, state, current_sample, prediction, timestep, prev_timestep)

    {state, prev_sample}
  end

  defnp step_plms(scheduler, state, sample, prediction) do
    # See Equation (12)

    state = store_prediction(state, prediction)

    prediction =
      (55 * elem(state.recent_prediction, 0) - 59 * elem(state.recent_prediction, 1) +
         37 * elem(state.recent_prediction, 2) - 9 * elem(state.recent_prediction, 3)) / 24

    timestep = state.timesteps[state.iteration]
    prev_timestep = timestep - state.timestep_gap

    prev_sample = prev_sample(scheduler, state, sample, prediction, timestep, prev_timestep)

    {state, prev_sample}
  end

  defnp prev_sample(scheduler, state, sample, prediction, timestep, prev_timestep) do
    # See Equation (11)

    alpha_bar_t = state.alpha_bars[timestep]

    alpha_bar_t_prev =
      if prev_timestep >= 0 do
        state.alpha_bars[prev_timestep]
      else
        case scheduler.alpha_clip_strategy do
          :one -> 1.0
          :alpha_zero -> state.alpha_bars[0]
        end
      end

    sample_coeff = (alpha_bar_t_prev / alpha_bar_t) ** 0.5

    noise_coeff = alpha_bar_t_prev - alpha_bar_t

    noise_denom_coeff =
      alpha_bar_t * (1 - alpha_bar_t_prev) ** 0.5 +
        (alpha_bar_t * (1 - alpha_bar_t) * alpha_bar_t_prev) ** 0.5

    noise =
      case scheduler.prediction_type do
        :noise ->
          prediction

        :angular_velocity ->
          Nx.sqrt(alpha_bar_t) * prediction + Nx.sqrt(1 - alpha_bar_t) * sample
      end

    sample_coeff * sample - noise_coeff * noise / noise_denom_coeff
  end

  deftransformp store_prediction(state, prediction) do
    recent_prediction =
      state.recent_prediction
      |> Tuple.delete_at(tuple_size(state.recent_prediction) - 1)
      |> Tuple.insert_at(0, prediction)

    %{state | recent_prediction: recent_prediction}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(scheduler, data) do
      import Bumblebee.Shared.Converters

      opts =
        convert!(data,
          num_train_steps: {"num_train_timesteps", number()},
          beta_schedule: {
            "beta_schedule",
            mapping(%{
              "linear" => :linear,
              "scaled_linear" => :quadratic,
              "squaredcos_cap_v2" => :squared_cosine
            })
          },
          beta_start: {"beta_start", number()},
          beta_end: {"beta_end", number()},
          prediction_type:
            {"prediction_type",
             mapping(%{"epsilon" => :noise, "v_prediction" => :angular_velocity})},
          alpha_clip_strategy: {
            "set_alpha_to_one",
            mapping(%{true => :one, false => :alpha_zero})
          },
          timesteps_offset: {"steps_offset", number()},
          reduce_warmup: {"skip_prk_steps", boolean()}
        )

      @for.config(scheduler, opts)
    end
  end
end
