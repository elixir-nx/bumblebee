Application.put_env(:sample, PhoenixDemo.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 8080],
  server: true,
  live_view: [signing_salt: "bumblebee"],
  secret_key_base: String.duplicate("b", 64),
  pubsub_server: PhoenixDemo.PubSub
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
      const SAMPLING_RATE = 16_000;

      const Microphone = {
        mounted() {
          this.mediaRecorder = null;

          this.el.addEventListener("mousedown", (event) => {
            this.startRecording();
          });

          this.el.addEventListener("mouseup", (event) => {
            this.stopRecording();
          });
        },

        startRecording() {
          this.audioChunks = [];

          navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
            this.mediaRecorder = new MediaRecorder(stream);

            this.mediaRecorder.addEventListener("dataavailable", (event) => {
              if (event.data.size > 0) {
                this.audioChunks.push(event.data);
              }
            });

            this.mediaRecorder.start();
          });
        },

        stopRecording() {
          this.mediaRecorder.addEventListener("stop", (event) => {
            if (this.audioChunks.length === 0) return;

            const audioBlob = new Blob(this.audioChunks);

            audioBlob.arrayBuffer().then((buffer) => {
              const context = new AudioContext({ sampleRate: SAMPLING_RATE });

              context.decodeAudioData(buffer, (audioBuffer) => {
                const pcmBuffer = this.audioBufferToPcm(audioBuffer);
                const buffer = this.convertEndianness32(
                  pcmBuffer,
                  this.getEndianness(),
                  this.el.dataset.endianness
                );
                this.upload("audio", [new Blob([buffer])]);
              });
            });
          });

          this.mediaRecorder.stop();
        },

        isRecording() {
          return this.mediaRecorder && this.mediaRecorder.state === "recording";
        },

        audioBufferToPcm(audioBuffer) {
          const numChannels = audioBuffer.numberOfChannels;
          const length = audioBuffer.length;

          const size = Float32Array.BYTES_PER_ELEMENT * length;
          const buffer = new ArrayBuffer(size);

          const pcmArray = new Float32Array(buffer);

          const channelDataBuffers = Array.from(
            { length: numChannels },
            (x, channel) => audioBuffer.getChannelData(channel)
          );

          // Average all channels upfront, so the PCM is always mono

          for (let i = 0; i < pcmArray.length; i++) {
            let sum = 0;

            for (let channel = 0; channel < numChannels; channel++) {
              sum += channelDataBuffers[channel][i];
            }

            pcmArray[i] = sum / numChannels;
          }

          return buffer;
        },

        convertEndianness32(buffer, from, to) {
          if (from === to) {
            return buffer;
          }

          // If the endianness differs, we swap bytes accordingly
          for (let i = 0; i < buffer.byteLength / 4; i++) {
            const b1 = buffer[i];
            const b2 = buffer[i + 1];
            const b3 = buffer[i + 2];
            const b4 = buffer[i + 3];
            buffer[i] = b4;
            buffer[i + 1] = b3;
            buffer[i + 2] = b2;
            buffer[i + 3] = b1;
          }

          return buffer;
        },

        getEndianness() {
          const buffer = new ArrayBuffer(2);
          const int16Array = new Uint16Array(buffer);
          const int8Array = new Uint8Array(buffer);

          int16Array[0] = 1;

          if (int8Array[0] === 1) {
            return "little";
          } else {
            return "big";
          }
        },
      };

      const hooks = { Microphone };

      const liveSocket = new window.LiveView.LiveSocket("/live", window.Phoenix.Socket, { hooks });
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
    {:ok,
     socket
     |> assign(transcription: nil)
     |> allow_upload(:audio, accept: :any, progress: &handle_progress/3, auto_upload: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="h-screen w-screen flex items-center justify-center antialiased">
      <div class="flex flex-col items-center w-1/2">
        <div class="mb-6 text-gray-600 text-lg">
          <h1>Press and hold</h1>
        </div>

        <button
          type="button"
          id="microphone"
          phx-hook="Microphone"
          data-endianness={System.endianness()}
          class="p-5 text-white bg-blue-700 rounded-full text-sm hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 active:bg-red-400 group"
        >
          <.microphone_icon class="w-8 h-8 group-active:animate-pulse" />
        </button>

        <form phx-change="noop" phx-submit="noop" class="hidden">
          <.live_file_input upload={@uploads.audio} />
        </form>

        <div class="mt-6 flex space-x-1.5 items-center text-gray-600 text-lg">
          <div>Transcription:</div>
          <.async_result :let={transcription} assign={@transcription} :if={@transcription}>
            <:loading>
              <.spinner />
            </:loading>
            <:failed :let={_reason}>
              <span>Oops, something went wrong!</span>
            </:failed>
            <span class="text-gray-900 font-medium"><%= transcription %></span>
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

  defp microphone_icon(assigns) do
    ~H"""
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      stroke-width="1.5"
      stroke="currentColor"
      class={@class}
    >
      <path
        stroke-linecap="round"
        stroke-linejoin="round"
        d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
      />
    </svg>
    """
  end

  defp handle_progress(:audio, entry, socket) when entry.done? do
    binary =
      consume_uploaded_entry(socket, entry, fn %{path: path} ->
        {:ok, File.read!(path)}
      end)

    # We always pre-process audio on the client into a single channel
    audio = Nx.from_binary(binary, :f32)

    socket =
      socket
      # Discard previous transcription so we show the loading state once more
      |> assign(:transcription, nil)
      |> assign_async(:transcription, fn ->
        output = Nx.Serving.batched_run(PhoenixDemo.Serving, audio)
        transcription = output.chunks |> Enum.map_join(& &1.text) |> String.trim()
        {:ok, %{transcription: transcription}}
      end)

    {:noreply, socket}
  end

  defp handle_progress(_name, _entry, socket), do: {:noreply, socket}

  @impl true
  def handle_event("noop", %{}, socket) do
    # We need phx-change and phx-submit on the form for live uploads,
    # but we make predictions immediately using :progress, so we just
    # ignore this event
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

{:ok, model_info} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})
{:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai/whisper-tiny"})

serving =
  Bumblebee.Audio.speech_to_text_whisper(model_info, featurizer, tokenizer, generation_config,
    compile: [batch_size: 4],
    defn_options: [compiler: EXLA]
  )

{:ok, _} =
  Supervisor.start_link(
    [
      {Phoenix.PubSub, name: PhoenixDemo.PubSub},
      {Nx.Serving, serving: serving, name: PhoenixDemo.Serving, batch_timeout: 100},
      PhoenixDemo.Endpoint
    ],
    strategy: :one_for_one
  )

Process.sleep(:infinity)
