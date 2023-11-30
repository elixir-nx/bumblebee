Application.put_env(:bumblebee, :progress_bar_enabled, false)

client = EXLA.Client.fetch!(:host)

exclude_multi_device = if client.device_count > 1, do: [], else: [:multi_device]

if client.device_count == 1 and System.schedulers_online() > 1 do
  IO.puts(
    "To run multi-device tests: XLA_FLAGS=--xla_force_host_platform_device_count=2 mix test"
  )
end

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda],
  rocm: [platform: :rocm],
  tpu: [platform: :tpu],
  other_host: [platform: :host, automatic_transfers: false]
)

Application.put_env(:exla, :preferred_clients, [:tpu, :cuda, :rocm, :other_host, :host])

Application.put_env(:nx, :default_backend, {EXLA.Backend, client: :host})

if System.fetch_env("BUMBLEBEE_OFFLINE") == :error do
  IO.puts("To run tests without hitting the network: BUMBLEBEE_OFFLINE=true mix test")
end

ExUnit.start(exclude: [:slow] ++ exclude_multi_device)
