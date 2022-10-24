Nx.global_default_backend(EXLA.Backend)

Application.put_env(:bumblebee, :progress_bar_enabled, false)

ExUnit.start(exclude: [:slow])
