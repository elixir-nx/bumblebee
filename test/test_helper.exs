Application.put_env(:nx, :default_backend, EXLA.Backend)
Application.put_env(:nx, :compiler, EXLA)

ExUnit.configure(exclude: [:slow])

ExUnit.start()
