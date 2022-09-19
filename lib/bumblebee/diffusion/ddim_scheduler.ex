defmodule Bumblebee.Diffusion.DdimScheduler do
  @moduledoc ~S"""
  Denoising diffusion implicit models (DDIMs).

  This sampling method was proposed as a follow up to the original
  denoising diffusion probabilistic models (DDPMs) in order to heavily
  reduce the number of steps during inference. DDPMs model the diffusion
  process as a Markov chain; DDIMs generalize this considering
  non-Markovian diffusion processes that lead to the same objective.
  This enables a reverse process with many less samples, as compared
  to DDPMs, while using the same denoising model.

  DDIMs were shown to be a simple variant of pseudo numerical methods
  for diffusion models (PNDMs), see `Bumblebee.Diffusion.Schedules.Pndm`
  and the corresponding paper for more details.

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

    * `:set_alpha_to_one` - each step $t$ uses the values of $\bar{\alpha}_t$
      and $\bar{\alpha}_{t-1}$, however for $t = 0$ there is no previous
      alpha. Setting this option to `true` implies $\bar{\alpha_}{t-1} = 1$,
      otherwise $\bar{\alpha}_{t-1} = \bar{\alpha}_0$. Defaults to `true`

    * `:steps_offset` - an offset added to the inference steps. You can
      use a combination of `offset: 1` and `set_alpha_to_one: false`,
      so that the last step $t = 1$ uses $\bar{\alpha}_1$ and $\bar{\alpha}_0$,
      as done in stable diffusion. Defaults to `0`

    * `:clip_sample` - whether to clip each predicted sample into $[-1, 1]$
      for numerical stability.. Defaults to `true`

    * `:use_clipped_model_output` - whether the noise (output of the
      denoising model) should be re-derived at each step based on the
      predicted original sample and the current sample. This technique
      is used in OpenAI GLIDE. Defaults to `false`

    * `:eta` - a weight for the noise added in a defnoising diffusion
      step. This scales the value of $\sigma_t$ in Equation (12) in
      the original paper, as per Equation (16). Defaults to `0.0`


  ## References

    * [Denoising diffusion implicit models](https://arxiv.org/pdf/2010.02502.pdf)

  """

  import Nx.Defn

  alias Bumblebee.Diffusion.SchedulerUtils

  @behaviour Bumblebee.Scheduler

  defstruct num_train_timesteps: 1000,
            beta_schedule: :linear,
            beta_start: 0.0001,
            beta_end: 0.02,
            set_alpha_to_one: true,
            steps_offset: 0,
            clip_sample: true,
            use_clipped_model_output: false,
            eta: 0.0

  @impl true
  def config(config, opts) do
    Bumblebee.Shared.put_config_attrs(config, opts)
  end

  @impl true
  def init(config, num_inference_timesteps, _sample_shape) do
    timesteps =
      SchedulerUtils.ddim_timesteps(
        config.num_train_timesteps,
        num_inference_timesteps,
        config.steps_offset
      )

    alpha_bars = init_parameters(config: config)

    state = %{
      timesteps: timesteps,
      timestep_gap: div(config.num_train_timesteps, num_inference_timesteps),
      alpha_bars: alpha_bars,
      iteration: 0
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

  @impl true
  deftransform step(scheduler, state, sample, noise) do
    do_step(scheduler, state, sample, noise)
  end

  defnp do_step(schedule \\ [], state, sample, noise) do
    # See Equation (12)

    # Note that in the paper alpha_t represents a cumulative product,
    # often denoted as alpha_t with a bar on top. We use an explicit
    # alpha_bart_t name for consistency

    timestep = state.timesteps[state.iteration]
    prev_timestep = timestep - state.timestep_gap

    alpha_bar_t = state.alpha_bars[timestep]

    alpha_bar_t_prev =
      if prev_timestep >= 0 do
        state.alpha_bars[prev_timestep]
      else
        if schedule.set_alpha_to_one, do: 1.0, else: state.alpha_bars[0]
      end

    pred_original_sample = (sample - Nx.sqrt(1 - alpha_bar_t) * noise) / Nx.sqrt(alpha_bar_t)

    pred_original_sample =
      if schedule.clip_sample do
        Nx.clip(pred_original_sample, -1, 1)
      else
        pred_original_sample
      end

    # See Equation (16)
    sigma_t =
      schedule.eta *
        Nx.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))

    noise =
      if schedule.use_clipped_model_output do
        # Re-derive the noise as in GLIDE
        (sample - Nx.sqrt(alpha_bar_t) * pred_original_sample) / Nx.sqrt(1 - alpha_bar_t)
      else
        noise
      end

    pred_sample_direction = Nx.sqrt(1 - alpha_bar_t_prev - Nx.power(sigma_t, 2)) * noise

    prev_sample = Nx.sqrt(alpha_bar_t_prev) * pred_original_sample + pred_sample_direction

    prev_sample =
      if schedule.eta > 0 do
        prev_sample + sigma_t * Nx.random_normal(prev_sample)
      else
        prev_sample
      end

    state = %{state | iteration: state.iteration + 1}

    {state, prev_sample}
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
