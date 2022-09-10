defmodule Bumblebee.Diffusion.Schedules.Utils do
  @moduledoc false

  import Nx.Defn

  @pi :math.pi()

  @doc """
  Returns a beta schedule of the given type.

  The supported types are:

    * `:linear` - a linear schedule from Ho et al. (https://arxiv.org/pdf/2006.11239.pdf)

    * `:scaled_linear` - a schedule specific to the latent diffusion models

    * `:squaredcos_cap_v2` - a cosine schedule from OpenAI GLIDE

  ## Options

    * `:linear_start` - start for the linear schedule. Defaults to `0.0001`

    * `:linear_end` - end for the liner schedule. Defaults to `0.02`

  """
  deftransform beta_schedule(type, num_timesteps, opts \\ []) do
    opts = Keyword.validate!(opts, linear_start: 0.0001, linear_end: 0.02)
    linear_start = opts[:linear_start]
    linear_end = opts[:linear_end]

    case type do
      :linear ->
        Bumblebee.Utils.Nx.linspace(linear_start, linear_end, steps: num_timesteps)

      :scaled_linear ->
        Bumblebee.Utils.Nx.linspace(Nx.sqrt(linear_start), Nx.sqrt(linear_end),
          steps: num_timesteps
        )
        |> Nx.power(2)

      :squaredcos_cap_v2 ->
        betas_for_alpha_bar(&squaredcos_cap_v2_alpha_bar/1, num_timesteps: num_timesteps)
    end
  end

  defnp squaredcos_cap_v2_alpha_bar(t) do
    Nx.cos((t + 0.008) / 1.008 * @pi / 2) ** 2
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
  deftransform ddim_timesteps(num_train_timesteps, num_inference_steps, offset) do
    step = div(num_train_timesteps, num_inference_steps)

    Nx.iota({num_inference_steps})
    |> Nx.multiply(step)
    |> Nx.add(offset)
    |> Nx.reverse()
  end
end
