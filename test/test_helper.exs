Nx.global_default_backend(EXLA.Backend)

ExUnit.start(exclude: [:slow])
