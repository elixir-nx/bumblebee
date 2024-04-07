Application.put_env(:sample, PhoenixDemo.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 8080],
  server: true,
  live_view: [signing_salt: "bumblebee"],
  secret_key_base: String.duplicate("b", 64)
)

Mix.install([
  {:plug_cowboy, "~> 2.6"},
  {:jason, "~> 1.4"},
  {:phoenix, "1.7.10"},
  {:phoenix_live_view, "0.20.1"},
  # Bumblebee and friends
  {:bumblebee, "~> 0.5.0"},
  {:nx, "~> 0.7.0"},
  {:exla, "~> 0.7.0"}
])

Application.put_env(:nx, :default_backend, EXLA.Backend)

defmodule PhoenixDemo.Layouts do
  use Phoenix.Component

  def render("live.html", assigns) do
    ~H"""
    <script src="https://cdn.jsdelivr.net/npm/phoenix@1.7.10/priv/static/phoenix.min.js">
    </script>
    <script
      src="https://cdn.jsdelivr.net/npm/phoenix_live_view@0.20.1/priv/static/phoenix_live_view.min.js"
    >
    </script>
    <script>
      const liveSocket = new window.LiveView.LiveSocket("/live", window.Phoenix.Socket);
      liveSocket.connect();
    </script>
    <script src="https://cdn.tailwindcss.com">
    </script>
    <%= @inner_content %>
    """
  end
end

defmodule PhoenixDemo.ErrorView do
  def render(_, _), do: "error"
end

defmodule PhoenixDemo.SampleLive do
  use Phoenix.LiveView, layout: {PhoenixDemo.Layouts, :live}

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, text: "", label: nil)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="h-screen w-screen flex items-center justify-center antialiased">
      <div class="flex flex-col h-1/2 w-1/2">
        <form phx-submit="predict" class="m-0 flex space-x-2">
          <input
            class="block w-full p-2.5 bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500"
            type="text"
            name="text"
            value={@text}
          />
          <button
            class="px-5 py-2.5 text-center mr-2 inline-flex items-center text-white bg-blue-700 font-medium rounded-lg text-sm hover:bg-blue-800 focus:ring-4 focus:ring-blue-300"
            type="submit"
            disabled={@label != nil and @label.loading != nil}
          >
            Predict
          </button>
        </form>
        <div class="mt-2 flex space-x-1.5 items-center text-gray-600 text-lg">
          <span>Emotion:</span>
          <.async_result :let={label} assign={@label} :if={@label}>
            <:loading>
              <.spinner />
            </:loading>
            <:failed :let={_reason}>
              <span>Oops, something went wrong!</span>
            </:failed>
            <span class="text-gray-900 font-medium"><%= label %></span>
          </.async_result>
        </div>
      </div>
    </div>
    """
  end

  defp spinner(assigns) do
    ~H"""
    <svg
      class="inline mr-2 w-4 h-4 text-gray-200 animate-spin fill-blue-600"
      viewBox="0 0 100 101"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
        fill="currentColor"
      />
      <path
        d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
        fill="currentFill"
      />
    </svg>
    """
  end

  @impl true
  def handle_event("predict", %{"text" => text}, socket) do
    socket =
      socket
      |> assign(:text, text)
      # Discard previous label so we show the loading state once more
      |> assign(:label, nil)
      |> assign_async(:label, fn ->
        output = Nx.Serving.batched_run(PhoenixDemo.Serving, text)
        %{predictions: [%{label: label}]} = output
        {:ok, %{label: label}}
      end)

    {:noreply, socket}
  end
end

defmodule PhoenixDemo.Router do
  use Phoenix.Router

  import Phoenix.LiveView.Router

  pipeline :browser do
    plug(:accepts, ["html"])
  end

  scope "/", PhoenixDemo do
    pipe_through(:browser)

    live("/", SampleLive, :index)
  end
end

defmodule PhoenixDemo.Endpoint do
  use Phoenix.Endpoint, otp_app: :sample

  socket("/live", Phoenix.LiveView.Socket)
  plug(PhoenixDemo.Router)
end

# Application startup

{:ok, model_info} = Bumblebee.load_model({:hf, "finiteautomata/bertweet-base-emotion-analysis"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "vinai/bertweet-base"})

serving =
  Bumblebee.Text.text_classification(model_info, tokenizer,
    top_k: 1,
    compile: [batch_size: 4, sequence_length: 100],
    defn_options: [compiler: EXLA]
  )

{:ok, _} =
  Supervisor.start_link(
    [
      {Nx.Serving, serving: serving, name: PhoenixDemo.Serving, batch_timeout: 100},
      PhoenixDemo.Endpoint
    ],
    strategy: :one_for_one
  )

Process.sleep(:infinity)
