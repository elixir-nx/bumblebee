defmodule Bumblebee.Diffusion.LcmScheduler do
  options = [
    num_train_steps: [
      default: 1000,
      doc: "the number of diffusion steps used to train the model"
    ],
    beta_schedule: [
      default: :quadratic,
      doc: """
      the beta schedule type, a mapping from a beta range to a sequence of betas for stepping the model.
      Either of `:linear`, `:quadratic`, or `:squared_cosine`
      """
    ],
    beta_start: [
      default: 0.00085,
      doc: "the start value for the beta schedule"
    ],
    beta_end: [
      default: 0.012,
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
    clip_denoised_sample: [
      default: false,
      doc: """
      whether to clip the predicted denoised sample ($x_0$ in Equation (12)) into $[-1, 1]$
      for numerical stability.
      """
    ],
    original_inference_steps: [
      default: 50,
      doc: ~S"""
      Default number of inference steps used to generate a linearly-spaced
      timestep schedule. from which we will ultimately take `num_inference_steps`
      evenly spaced timesteps to form the final timestep schedule.
      """
    ],
    timestep_scaling: [
      default: 10.0,
      doc: ~S"""
      Multiplier factor for timesteps. Used when calculating the consistency model
      boundary conditions `c_skip` and `c_out`. Increasing this will decrease
      the approximation error (although the approximation error at the default of
      `10.0` is already pretty small).
      """
    ],
    sigma_data: [
      default: 0.5,
      doc: ~S"""
      Used to calculate the scaling factors for denoising model output.
      """
    ]
  ]

  @moduledoc """
  Latent Consistency Model (LCM) sampling

  This sampling method uses classifier-free guidance and cycles of
  denoising and noise injection to improve sample quality.
  Although the maximum number of inference steps can be set to 50,
  steps of 2-4 with a guidance_scale of 1.0 - 2.0 generate better results.
  This scheduler does not support custom timesteps or img2img workflows.

  See Appendix B in the paper below for more details.

  ## Configuration

  #{Bumblebee.Shared.options_doc(options)}

  ## References

    * [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)

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
  def init(scheduler, num_steps, _sample_shape, opts \\ []) do
    opts = Keyword.validate!(opts, [:seed, :strength])

    seed = Keyword.get_lazy(opts, :seed, fn -> :erlang.system_time() end)
    strength = Keyword.get(opts, :strength, 1.0)

    timesteps =
      timesteps(
        num_steps,
        scheduler.original_inference_steps,
        scheduler.num_train_steps,
        strength
      )

    {alpha_bars, prng_key} = init_parameters(scheduler: scheduler, seed: seed)

    state =
      %{
        timesteps: timesteps,
        timestep_gap: div(scheduler.num_train_steps, num_steps),
        num_steps: num_steps,
        alpha_bars: alpha_bars,
        prng_key: prng_key,
        step_index: 0
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

    seed = opts[:seed]
    prng_key = Nx.Random.key(seed)

    betas =
      SchedulerUtils.beta_schedule(beta_schedule, num_train_steps,
        start: beta_start,
        end: beta_end
      )

    alphas = 1 - betas

    {Nx.cumulative_product(alphas), prng_key}
  end

  @impl true
  def step(scheduler, state, sample, prediction) do
    do_step(state, sample, prediction, scheduler: scheduler)
  end

  defnp do_step(state, sample, prediction, opts) do
    scheduler = opts[:scheduler]

    step_index = state.step_index
    prev_step_index = state.step_index + 1

    timestep = state.timesteps[step_index]

    prev_timestep =
      if prev_step_index < Nx.size(state.timesteps) do
        state.timesteps[prev_step_index]
      else
        timestep
      end

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

    {c_skip, c_out} =
      get_scalings_for_boundary_condition_discrete(timestep,
        timestep_scaling: scheduler.timestep_scaling,
        sigma_data: scheduler.sigma_data
      )

    pred_denoised_sample =
      case scheduler.prediction_type do
        :noise ->
          (sample - Nx.sqrt(beta_bar_t) * prediction) / Nx.sqrt(alpha_bar_t)

        :angular_velocity ->
          Nx.sqrt(alpha_bar_t) * sample - Nx.sqrt(beta_bar_t) * prediction
      end

    pred_denoised_sample =
      if scheduler.clip_denoised_sample do
        Nx.clip(pred_denoised_sample, -1, 1)
      else
        pred_denoised_sample
      end

    denoised = c_out * pred_denoised_sample + c_skip * sample

    # Noise is not used on the final timestep of the timestep schedule.
    # This also means that noise is not used for one-step sampling

    {prev_sample, next_key} =
      if step_index != state.num_steps - 1 do
        {noise, next_key} = Nx.Random.normal(state.prng_key, shape: prediction)
        out = Nx.sqrt(alpha_bar_t_prev) * denoised + Nx.sqrt(beta_bar_t_prev) * noise
        {out, next_key}
      else
        {denoised, state.prng_key}
      end

    state = %{
      state
      | step_index: step_index + 1,
        prng_key: next_key
    }

    {state, prev_sample}
  end

  defnp get_scalings_for_boundary_condition_discrete(timestep, opts) do
    timestep_scaling = opts[:timestep_scaling]
    sigma_data = opts[:sigma_data]

    scaled_timestep = timestep * timestep_scaling

    c_skip = Nx.pow(sigma_data, 2) / (Nx.pow(scaled_timestep, 2) + Nx.pow(sigma_data, 2))
    c_out = scaled_timestep / Nx.sqrt(Nx.pow(scaled_timestep, 2) + Nx.pow(sigma_data, 2))
    {c_skip, c_out}
  end

  deftransformp timesteps(
                  num_steps,
                  original_inference_steps,
                  num_train_timesteps,
                  strength
                ) do
    k = div(num_train_timesteps, original_inference_steps)

    lcm_origin_timesteps =
      {round(original_inference_steps * strength)}
      |> Nx.iota()
      |> Nx.add(1)
      |> Nx.multiply(k)
      |> Nx.subtract(1)

    if num_steps > num_train_timesteps do
      raise ArgumentError,
            "the number of inference steps needs to be less than the original training timesteps (#{num_train_timesteps}), got: #{num_steps}"
    end

    skipping_step = div(Nx.size(lcm_origin_timesteps), num_steps)

    if skipping_step < 1 do
      raise ArgumentError,
            "the steps between lcm_origin_timesteps needs to be a positive integer, " <>
              "change inference steps (#{num_steps}) to be" <>
              " less than original_inference_steps (#{original_inference_steps})" <>
              " * strength (#{strength})"
    end

    if num_steps > original_inference_steps do
      raise ArgumentError,
            "the number of inference steps (#{num_steps}) cannot be greater than" <>
              " original_inference_steps (#{original_inference_steps})"
    end

    lcm_origin_timesteps = Nx.reverse(lcm_origin_timesteps)

    inference_indices =
      Bumblebee.Utils.Nx.linspace(0, Nx.size(lcm_origin_timesteps) - 1, steps: num_steps)

    inference_indices = Nx.floor(inference_indices) |> Nx.as_type({:s, 64})

    Nx.take(lcm_origin_timesteps, inference_indices)
  end
 end
