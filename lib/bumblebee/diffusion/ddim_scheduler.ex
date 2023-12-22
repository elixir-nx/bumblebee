defmodule Bumblebee.Diffusion.DdimScheduler do
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
      default: :one,
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
    clip_denoised_sample: [
      default: true,
      doc: """
      whether to clip the predicted denoised sample ($x_0$ in Equation (12)) into $[-1, 1]$
      for numerical stability
      """
    ],
    rederive_noise: [
      default: false,
      doc: """
      whether the noise (output of the denoising model) should be re-derived at each step based on the
      predicted denoised sample ($x_0$) and the current sample. This technique is used in OpenAI GLIDE
      """
    ],
    eta: [
      default: 0.0,
      doc: """
      a weight for the noise added in a denoising diffusion step. This scales the value of $\\sigma_t$
      in Equation (12) in the original paper, as per Equation (16)
      """
    ]
  ]

  @moduledoc """
  Denoising diffusion implicit models (DDIMs).

  This sampling method was proposed as a follow up to the original
  denoising diffusion probabilistic models (DDPMs) in order to heavily
  reduce the number of steps during inference. DDPMs model the diffusion
  process as a Markov chain; DDIMs generalize this considering
  non-Markovian diffusion processes that lead to the same objective.
  This enables a reverse process with many less samples, as compared
  to DDPMs, while using the same denoising model.

  DDIMs were shown to be a simple variant of pseudo numerical methods
  for diffusion models (PNDMs), see `Bumblebee.Diffusion.PndmScheduler`
  and the corresponding paper for more details.

  ## Configuration

  #{Bumblebee.Shared.options_doc(options)}

  ## References

    * [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

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
  def init(scheduler, num_steps, _sample_template, prng_key) do
    timesteps =
      SchedulerUtils.ddim_timesteps(
        scheduler.num_train_steps,
        num_steps,
        scheduler.timesteps_offset
      )

    {alpha_bars} = init_parameters(scheduler: scheduler)

    state = %{
      timesteps: timesteps,
      timestep_gap: div(scheduler.num_train_steps, num_steps),
      alpha_bars: alpha_bars,
      iteration: 0,
      prng_key: prng_key
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

    {Nx.cumulative_product(alphas)}
  end

  @impl true
  def step(scheduler, state, sample, prediction) do
    do_step(state, sample, prediction, scheduler: scheduler)
  end

  defnp do_step(state, sample, prediction, opts) do
    scheduler = opts[:scheduler]

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
        case scheduler.alpha_clip_strategy do
          :one -> 1.0
          :alpha_zero -> state.alpha_bars[0]
        end
      end

    beta_bar_t = 1 - alpha_bar_t
    beta_bar_t_prev = 1 - alpha_bar_t_prev

    {pred_denoised_sample, noise} =
      case scheduler.prediction_type do
        :noise ->
          pred_denoised_sample =
            (sample - Nx.sqrt(beta_bar_t) * prediction) / Nx.sqrt(alpha_bar_t)

          {pred_denoised_sample, prediction}

        :angular_velocity ->
          pred_denoised_sample =
            Nx.sqrt(alpha_bar_t) * sample - Nx.sqrt(beta_bar_t) * prediction

          noise = Nx.sqrt(alpha_bar_t) * prediction + Nx.sqrt(beta_bar_t) * sample
          {pred_denoised_sample, noise}
      end

    pred_denoised_sample =
      if scheduler.clip_denoised_sample do
        Nx.clip(pred_denoised_sample, -1, 1)
      else
        pred_denoised_sample
      end

    # See Equation (16)
    sigma_t =
      scheduler.eta *
        Nx.sqrt(beta_bar_t_prev / beta_bar_t * (1 - alpha_bar_t / alpha_bar_t_prev))

    noise =
      if scheduler.rederive_noise do
        # Re-derive the noise as in GLIDE
        (sample - Nx.sqrt(alpha_bar_t) * pred_denoised_sample) / Nx.sqrt(beta_bar_t)
      else
        noise
      end

    pred_sample_direction = Nx.sqrt(beta_bar_t_prev - Nx.pow(sigma_t, 2)) * noise

    prev_sample = Nx.sqrt(alpha_bar_t_prev) * pred_denoised_sample + pred_sample_direction

    {prev_sample, next_key} =
      if scheduler.eta > 0 do
        {rand, next_key} = Nx.Random.normal(state.prng_key, shape: Nx.shape(prev_sample))
        out = prev_sample + sigma_t * rand
        {out, next_key}
      else
        {prev_sample, state.prng_key}
      end

    state = %{state | iteration: state.iteration + 1, prng_key: next_key}

    {state, prev_sample}
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
          clip_denoised_sample: {"clip_sample", boolean()},
          rederive_noise: {"use_clipped_model_output", boolean()}
        )

      @for.config(scheduler, opts)
    end
  end
end
