defmodule Bumblebee.Scheduler do
  @moduledoc """
  An interface for configuring and using schedulers.

  A scheduler defines a sampling method, usually used for multi-step
  denoising process, as in stable diffusion.

  Every module implementing this behaviour is expected to also define
  a configuration struct.

  ## Context

  Imagine a denoising model trained in 1000 steps. During training,
  we take some original data and add random noise 1000 times, this
  way we obtain 1000 steps with increasing level of noise. Then, the
  model learns to predict noise at each timestep, given data at that
  step (sample) and the timestep.

  Once such model is trained, we can obtain brand new data (such as
  image) by generating random data and denoising it with our model in
  1000 steps.

  Doing 1000 forward passes of the model for a single generation can
  be expensive, hence multiple methods have been developed to reduce
  the number of steps during denoising, with no changes to the model.

  Each method specifies a subset of the original timesteps, at each
  timestep we need to do a forward pass of the model (or possibly a
  few), then the method extrapolates the sample to the next selected
  timestep, possibly skipping a lot of timesteps in between.

  ## Note on wording

  Throughout the docs and APIs the word "steps" refers to diffusion
  steps, whereas "timesteps" is more specific and refers to the exact
  values $t$ (points in time).
  """

  @type t :: Bumblebee.Configurable.t()

  @type state :: Nx.Container.t()

  @doc """
  Initializes state for a new scheduler loop.

  Returns a pair of `{state, timesteps}`, where `state` is an opaque
  `Nx.Container` and `timesteps` is a tensor with the subsequent
  timesteps for model forward pass.
  """
  @callback init(
              t(),
              num_steps :: pos_integer(),
              sample_template :: Nx.Tensor.t(),
              prng_key :: Nx.Tensor.t()
            ) :: {state :: map(), timesteps :: Nx.Tensor.t()}

  @doc """
  Predicts sample at the previous timestep.

  Takes the current `sample` and `prediction` (usually noise) returned
  by the model at the current timestep. Returns `{state, prev_sample}`,
  where `state` is the updated state and `prev_sample` is the predicted
  sample at the previous timestep.
  """
  @callback step(
              t(),
              state(),
              sample :: Nx.Tensor.t(),
              prediction :: Nx.Tensor.t()
            ) :: {state :: map(), prev_sample :: Nx.Tensor.t()}
end
