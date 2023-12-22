defmodule Bumblebee.Diffusion.SchedulerUtils do
  @moduledoc false

  import Nx.Defn

  @pi :math.pi()

  @doc """
  Returns a beta schedule of the given type.

  The supported types are:

    * `:linear` - a linear schedule from Ho et al. (https://arxiv.org/pdf/2006.11239.pdf)

    * `:quadratic` - a quadratic schedule specific to the latent diffusion models

    * `:squared_cosine` - a cosine schedule from Nichol et al. (https://arxiv.org/pdf/2102.09672.pdf),
      used in OpenAI GLIDE

  ## Options

    * `:start` - start for the linear and quadratic schedules. Defaults to `0.0001`

    * `:end` - end for the linear and quadratic schedules. Defaults to `0.02`

  """
  deftransform beta_schedule(type, num_timesteps, opts \\ []) do
    opts = Keyword.validate!(opts, start: 0.0001, end: 0.02)
    beta_start = opts[:start]
    beta_end = opts[:end]

    case type do
      :linear ->
        Nx.linspace(beta_start, beta_end, n: num_timesteps)

      :quadratic ->
        Nx.linspace(Nx.sqrt(beta_start), Nx.sqrt(beta_end), n: num_timesteps) |> Nx.pow(2)

      :squared_cosine ->
        betas_for_alpha_bar(&squared_cosine_alpha_bar/1, num_timesteps: num_timesteps)
    end
  end

  defnp squared_cosine_alpha_bar(t) do
    s = 0.008
    Nx.cos((t + s) / (1 + s) * @pi / 2) ** 2
  end

  # Creates a beta schedule that discretizes the given alpha_t_bar function,
  # which defines the cumulative product of (1 - beta) over time t in [0, 1].
  defnp betas_for_alpha_bar(alpha_t_bar_fun, opts \\ []) do
    opts = keyword!(opts, [:num_timesteps, max_beta: 0.999])
    num_timesteps = opts[:num_timesteps]
    max_beta = opts[:max_beta]

    i = Nx.iota({num_timesteps})
    t1 = i / num_timesteps
    t2 = (i + 1) / num_timesteps
    beta = 1 - alpha_t_bar_fun.(t2) / alpha_t_bar_fun.(t1)
    min(beta, max_beta)
  end

  @doc """
  Returns evenly spaced timesteps as used in the DDIM schedule.
  """
  deftransform ddim_timesteps(num_train_steps, num_steps, offset) do
    timestep_gap = div(num_train_steps, num_steps)

    Nx.iota({num_steps})
    |> Nx.multiply(timestep_gap)
    |> Nx.add(offset)
    |> Nx.reverse()
  end
end
