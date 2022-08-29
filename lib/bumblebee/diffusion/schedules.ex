defmodule Bumblebee.Diffusion.Schedules do
  import Nx.Defn

  # TODO: It migth be better to take a module appraoch
  # per scheduler

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
