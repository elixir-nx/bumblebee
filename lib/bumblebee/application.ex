defmodule Bumblebee.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    Bumblebee.Utils.HTTP.start_inets_profile()

    children = []
    opts = [strategy: :one_for_one, name: Bumblebee.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @impl true
  def stop(_state) do
    Bumblebee.Utils.HTTP.stop_inets_profile()
  end
end
