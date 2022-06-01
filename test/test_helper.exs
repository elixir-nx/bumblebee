Nx.default_backend(EXLA.Backend)

ExUnit.configure(exclude: [:slow])

ExUnit.start()
