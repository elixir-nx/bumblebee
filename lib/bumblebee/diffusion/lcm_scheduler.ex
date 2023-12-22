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
      for numerical stability
      """
    ],
    num_original_steps: [
      default: 50,
      doc: ~S"""
      the number of denoising steps used during Latent Consistency Distillation (LCD).
      The LCD procedure distills a base diffusion model, but instead of sampling all
      `:num_train_steps` it skips steps and uses another scheduler accordingly. See
      Section 4.3
      """
    ],
    boundary_condition_timestep_scale: [
      default: 10.0,
      doc: ~S"""
      the scaling factor used in the consistency function coefficients. In the original
      LCM implementation the authors use the formulation
      $$
      c_{skip}(t) = \frac{\sigma_{data}^2}{(st)^2 + \sigma_{data}^2}, \quad
      c_{out}(t) = \frac{st}{\sqrt{(st)^2 + \sigma_{data}^2}}
      $$
      where $\sigma_{data} = 0.5$ and $s$ is the scaling factor. Increasing the scaling
      factor will decrease approximation error, although the approximation error at the
      default of `10.0` is already pretty small
      """
    ]
  ]

  @moduledoc """
  Latent Consistency Model (LCM) sampling.

  This sampling method should be used in combination with LCM. LCM is
  a model distilled from a regular diffusion model to predict the
  final denoised sample in a single step. The sample quality can be
  improved by alternating a couple denoising and noise injection
  steps (multi-step sampling), as per Appendix B.

  ## Configuration

  #{Bumblebee.Shared.options_doc(options)}

  ## References

    * [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)
    * [Consistency Models](https://arxiv.org/pdf/2303.01469.pdf)

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
  def init(scheduler, num_steps, _sample_template, prng_key, opts \\ []) do
    opts = Keyword.validate!(opts, [:strength])

    strength = Keyword.get(opts, :strength, 1.0)

    timesteps =
      timesteps(num_steps, scheduler.num_original_steps, scheduler.num_train_steps, strength)

    {alpha_bars} = init_parameters(scheduler: scheduler)

    state = %{
      timesteps: timesteps,
      alpha_bars: alpha_bars,
      iteration: 0,
      prng_key: prng_key
    }

    {state, timesteps}
  end

  deftransformp timesteps(num_steps, num_original_steps, num_train_steps, strength) do
    skipping_step_k = div(num_train_steps, num_original_steps)

    # Original steps used during Latent Consistency Distillation
    original_timesteps =
      {floor(num_original_steps * strength)}
      |> Nx.iota()
      |> Nx.add(1)
      |> Nx.multiply(skipping_step_k)
      |> Nx.subtract(1)

    if num_steps > num_train_steps do
      raise ArgumentError,
            "expected the number of steps to be less or equal to the number of" <>
              " training steps (#{num_train_steps}), got: #{num_steps}"
    end

    if num_steps > num_original_steps do
      raise ArgumentError,
            "expected the number of steps to be less or equal to the number of" <>
              " original steps (#{num_original_steps}), got: #{num_steps}"
    end

    if num_steps > Nx.size(original_timesteps) do
      raise ArgumentError,
            "expected the number of steps to be less or equal to num_original_steps * strength" <>
              " (#{num_original_steps} * #{strength}). Either reduce the number of steps or" <>
              "increase the strength"
    end

    # We select evenly spaced indices from the original timesteps.
    # See the discussion in https://github.com/huggingface/diffusers/pull/5836
    indices =
      Nx.linspace(0, Nx.size(original_timesteps), n: num_steps, endpoint: false)
      |> Nx.floor()
      |> Nx.as_type(:s64)

    original_timesteps
    |> Nx.reverse()
    |> Nx.take(indices)
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

    step_index = state.iteration
    prev_step_index = state.iteration + 1

    timestep = state.timesteps[step_index]

    prev_timestep =
      if prev_step_index < Nx.size(state.timesteps) do
        state.timesteps[prev_step_index]
      else
        timestep
      end

    # Note that in the paper alpha_bar_t is denoted as a(t) and
    # beta_bar_t is denoted as sigma(t)^2

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

    # See Appendix D
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

    {c_skip, c_out} =
      consistency_model_coefficients(timestep,
        boundary_condition_timestep_scale: scheduler.boundary_condition_timestep_scale
      )

    # See Equation (9)
    denoised_sample = c_skip * sample + c_out * pred_denoised_sample

    # See Appendix B
    #
    # We insert additional noise after each but last step. This also
    # means no noise is used for one-step sampling
    {prev_sample, next_key} =
      if state.iteration < Nx.size(state.timesteps) - 1 do
        {rand, next_key} = Nx.Random.normal(state.prng_key, shape: Nx.shape(denoised_sample))
        out = Nx.sqrt(alpha_bar_t_prev) * denoised_sample + Nx.sqrt(beta_bar_t_prev) * rand
        {out, next_key}
      else
        {denoised_sample, state.prng_key}
      end

    state = %{state | iteration: state.iteration + 1, prng_key: next_key}

    {state, prev_sample}
  end

  defnp consistency_model_coefficients(timestep, opts) do
    # See Appendix C in https://arxiv.org/pdf/2303.01469.pdf
    #
    # Note that LCM authors use different coefficients for the
    # consistency function than the original CM paper. In their
    # formulation the timestep is scaled by a constant factor.

    boundary_condition_timestep_scale = opts[:boundary_condition_timestep_scale]
    sigma_data = 0.5

    scaled_timestep = timestep * boundary_condition_timestep_scale

    c_skip = sigma_data ** 2 / (scaled_timestep ** 2 + sigma_data ** 2)
    c_out = scaled_timestep / Nx.sqrt(scaled_timestep ** 2 + sigma_data ** 2)

    {c_skip, c_out}
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
          clip_denoised_sample: {"clip_sample", boolean()},
          num_original_steps: {"original_inference_steps", number()},
          boundary_condition_timestep_scale: {"timestep_scaling", number()}
        )

      @for.config(scheduler, opts)
    end
  end
end
