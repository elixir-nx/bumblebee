Mix.install([
  {:phoenix_playground, "~> 0.1.7"},
  {:bumblebee, "~> 0.6.0"},
  {:nx, "~> 0.9.0"},
  {:exla, "~> 0.9.0"}
])

Application.put_env(:nx, :default_backend, EXLA.Backend)

defmodule DemoLive do
  use Phoenix.LiveView

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(label: nil)
     |> allow_upload(:image, accept: :any, progress: &handle_progress/3, auto_upload: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <script src="https://cdn.tailwindcss.com">
    </script>

    <div class="h-screen w-screen flex items-center justify-center antialiased">
      <div class="flex flex-col items-center w-1/2">
        <form class="m-0 flex flex-col items-center space-y-2" phx-change="noop" phx-submit="noop">
          <.image_input id="image" upload={@uploads.image} height={224} width={224} />
        </form>
        <div class="mt-6 flex space-x-1.5 items-center text-gray-600 text-lg">
          <span>Label:</span>
          <.async_result :let={label} :if={@label} assign={@label}>
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

      <script>
        const DROP_CLASSES = ["bg-blue-100", "border-blue-300"];

        window.hooks.ImageInput = {
          mounted() {
            this.props = {
              height: parseInt(this.el.dataset.height),
              width: parseInt(this.el.dataset.width),
            };

            this.inputEl = this.el.querySelector(`[data-el-input]`);
            this.previewEl = this.el.querySelector(`[data-el-preview]`);

            // File selection

            this.el.addEventListener("click", (event) => {
              this.inputEl.click();
            });

            this.inputEl.addEventListener("change", (event) => {
              const [file] = event.target.files;
              file && this.loadFile(file);
            });

            // Drag and drop

            this.el.addEventListener("dragover", (event) => {
              event.stopPropagation();
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
            });

            this.el.addEventListener("drop", (event) => {
              event.stopPropagation();
              event.preventDefault();
              const [file] = event.dataTransfer.files;
              file && this.loadFile(file);
            });

            this.el.addEventListener("dragenter", (event) => {
              this.el.classList.add(...DROP_CLASSES);
            });

            this.el.addEventListener("dragleave", (event) => {
              if (!this.el.contains(event.relatedTarget)) {
                this.el.classList.remove(...DROP_CLASSES);
              }
            });

            this.el.addEventListener("drop", (event) => {
              this.el.classList.remove(...DROP_CLASSES);
            });
          },

          loadFile(file) {
            const reader = new FileReader();

            reader.onload = (readerEvent) => {
              const imgEl = document.createElement("img");

              imgEl.addEventListener("load", (loadEvent) => {
                this.setPreview(imgEl);

                const canvas = this.toCanvas(imgEl);
                const blob = this.canvasToBlob(canvas);
                this.upload("image", [blob]);
              });

              imgEl.src = readerEvent.target.result;
            };

            reader.readAsDataURL(file);
          },

          setPreview(imgEl) {
            // Keep the original image size intact
            const previewImgEl = imgEl.cloneNode();
            previewImgEl.style.maxHeight = "100%";
            this.previewEl.replaceChildren(previewImgEl);
          },

          toCanvas(imgEl) {
            // We resize the image, such that it fits in the configured
            // height x width, but keeping the aspect ratio. We could
            // also easily crop, pad or squash the image, if desired

            const { width, height } = imgEl;
            const { width: boundWidth, height: boundHeight } = this.props;

            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");

            const widthScale = boundWidth / width;
            const heightScale = boundHeight / height;
            const scale = Math.min(widthScale, heightScale);

            const scaledWidth = Math.round(width * scale);
            const scaledHeight = Math.round(height * scale);

            canvas.width = scaledWidth;
            canvas.height = scaledHeight;

            ctx.drawImage(imgEl, 0, 0, width, height, 0, 0, scaledWidth, scaledHeight);

            return canvas;
          },

          canvasToBlob(canvas) {
            const imageData = canvas
              .getContext("2d")
              .getImageData(0, 0, canvas.width, canvas.height);

            const buffer = this.imageDataToRGBBuffer(imageData);

            const meta = new ArrayBuffer(8);
            const view = new DataView(meta);
            view.setUint32(0, canvas.height, false);
            view.setUint32(4, canvas.width, false);

            return new Blob([meta, buffer], { type: "application/octet-stream" });
          },

          imageDataToRGBBuffer(imageData) {
            const pixelCount = imageData.width * imageData.height;
            const bytes = new Uint8ClampedArray(pixelCount * 3);

            for (let i = 0; i < pixelCount; i++) {
              bytes[i * 3] = imageData.data[i * 4];
              bytes[i * 3 + 1] = imageData.data[i * 4 + 1];
              bytes[i * 3 + 2] = imageData.data[i * 4 + 2];
            }

            return bytes.buffer;
          },
        };
      </script>
    </div>
    """
  end

  defp image_input(assigns) do
    ~H"""
    <div
      id={"#{@id}-root"}
      class="inline-flex p-4 border-2 border-dashed border-gray-200 rounded-lg cursor-pointer"
      phx-hook="ImageInput"
      data-height={@height}
      data-width={@width}
    >
      <.live_file_input upload={@upload} class="hidden" />
      <input type="file" data-el-input class="hidden" />
      <div
        class="h-[300px] w-[300px] flex items-center justify-center"
        id={"#{@id}-preview"}
        phx-update="ignore"
        data-el-preview
      >
        <div class="text-gray-500 text-center">
          Drag an image file here or click to open file browser
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

  def handle_progress(:image, entry, socket) when entry.done? do
    binary =
      consume_uploaded_entry(socket, entry, fn %{path: path} ->
        {:ok, File.read!(path)}
      end)

    image = decode_as_tensor(binary)

    socket =
      socket
      # Discard previous label so we show the loading state once more
      |> assign(:label, nil)
      |> assign_async(:label, fn ->
        output = Nx.Serving.batched_run(Demo.Serving, image)
        %{predictions: [%{label: label}]} = output
        {:ok, %{label: label}}
      end)

    {:noreply, socket}
  end

  def handle_progress(_name, _entry, socket), do: {:noreply, socket}

  defp decode_as_tensor(<<height::32-integer, width::32-integer, data::binary>>) do
    data |> Nx.from_binary(:u8) |> Nx.reshape({height, width, 3})
  end

  @impl true
  def handle_event("noop", %{}, socket) do
    # We need phx-change and phx-submit on the form for live uploads,
    # but we make predictions immediately using :progress, so we just
    # ignore this event
    {:noreply, socket}
  end
end

# Application startup

{:ok, model_info} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, "microsoft/resnet-50"})

serving =
  Bumblebee.Vision.image_classification(model_info, featurizer,
    top_k: 1,
    compile: [batch_size: 4],
    defn_options: [
      compiler: EXLA,
      cache: Path.join(System.tmp_dir!(), "bumblebee_examples/image_classification")
    ]
  )

Nx.Serving.start_link(serving: serving, name: Demo.Serving, batch_timeout: 100)

PhoenixPlayground.start(live: DemoLive, port: 8080)
