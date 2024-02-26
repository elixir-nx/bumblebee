# Phoenix LiveView examples

This directory contains minimal, single-file LiveView applications, which showcase how to integrate Bumblebee as part of the application.

## Running

Each example is fully contained in a single script (including all the dependencies), so you can run it as:

```shell
elixir example.exs
```

## How

Integrating a Bumblebee task is straightforward and boils down to two steps:

  1. On application startup we load the necessary models and preprocessors, build an instance of `Nx.Serving` and start it under the application supervision tree.

  2. Whenever we want to make a prediction for a user input, we use `Nx.Serving.batched_run/2` with said input, usually in a `Task` to avoid blocking other interactions. It is important to use `Nx.Serving.batched_run/2` as it automatically batches requests from concurrent users and distributes the results back to them transparently.

## Tips

### Deployment considerations

The examples in this directory download the model data directly from HuggingFace the first time the application starts. To avoid downloading the model data during deployments, you have two options.

#### 1. Explicit local versioning

One option is to version those files alongside your repository (perhaps using [Git LFS](https://git-lfs.github.com/)) or use an object storage service. Then, you'd fetch those files into a local directory as part of development/deployment.

In such scenarios, you must change the calls from `Bumblebee.load_xyz({:hf, "microsoft/resnet"})` to `Bumblebee.load_xyz({:local, "/path/to/model"})` and so on.

#### 2. Cached from Hugging Face

Bumblebee can download and cache data from both public and private Hugging Face repositories. You can control the cache directory by setting `BUMBLEBEE_CACHE_DIR`. Therefore, one option during deployment is to set the `BUMBLEBEE_CACHE_DIR` to a directory within your application. If using Docker, you must then include said directory in your application and make sure that `BUMBLEBEE_CACHE_DIR` points to it.

When using Docker multi-stage build, you can populate the cache in the build stage. For example, if your application has a module named `MyApp.Servings` with several Bumblebee models, you could define a function called `load_all/0` that loads all relevant information:

```elixir
  def load_all do
    Bumblebee.load_xyz({:hf, "microsoft/resnet"})
    Bumblebee.load_xyz({:hf, "foo/bar/baz"})
  end
```

Then your Dockerfile calls either `RUN mix eval 'MyApp.Servings.load_all()'` in your Mix project root or `RUN bin/my_app eval 'MyApp.Servings.load_all()'` in your release root to write all relevant information to the cache directory. The last step is to copy the contents of `BUMBLEBEE_CACHE_DIR` to the final image.

It is also recommended to set `BUMBLEBEE_OFFLINE` to `true` in the final image to make sure the models are always loaded from the cache.

### Configuring Nx

We currently recommend [EXLA](https://hexdocs.pm/exla/EXLA.html) to compile the numerical computations.

There are generally two types of computations you run with Nx:

  1. One-off operations, such as calling `Nx.sum(data)`, oftentimes used when preparing data for the model. Those operations are delegated to the default backend compiled one-by-one.

  2. Large defn computations, such as running a neural network model. Those are usually compiled as a whole explicitly using the configured compiler.

EXLA allows only a single computation per device to run at the same time, so if a GPU is available we want it to only run computations falling under the 2. type.

To achieve that, you can configure your default backend (used for 1.) to always use the CPU:

```elixir
config :nx, :default_backend, {EXLA.Backend, client: :host}
```

Then, for any expensive computations you can use [`Nx.Defn.compile/3`](https://hexdocs.pm/nx/Nx.Defn.html#compile/3) (or [`Axon.compile/4`](https://hexdocs.pm/axon/Axon.html#compile/4)) passing `compiler: EXLA` as an option. When you use a Bumblebee serving the compilation is handled for you, just make sure to pass `:compile` and `defn_options: [compiler: EXLA]` when creating the serving.

There's a final important detail related to parameters. With the above configuration, a model will run on the GPU, however parameters will be loaded onto the CPU (due to the default backend), so they will need to be copied onto the GPU every time. To avoid that, you can load the parameters onto the GPU directly using `Bumblebee.load_model(..., backend: EXLA.Backend)`.

When building the Bumblebee serving, make sure to specify the compiler and `:compile` shapes, so that the computation is compiled upfront when the serving boots.

```elixir
serving =
  Bumblebee.Text.text_embedding(model_info, tokenizer,
    compile: [batch_size: 1, sequence_length: 512],
    defn_options: [compiler: EXLA]
  )
```

### User images

When working with user-given images, the most trivial approach would be to just upload an image as is, in a format like PNG or JPEG. However, this approach has two downsides:

  1. In most cases full-resolution images are not necessary, because Neural Networks work on much smaller inputs. This means that a lot of data is unnecessarily uploaded over the network and that the server needs to do more work downsizing a potentially large image.

  2. Decoding an image format like PNG or JPEG requires an additional package and again, puts more work on the server.

Both of these downsides can be avoided by moving all the work to the client. Specifically, when the user selects an image, we can resize it to a much smaller version and decode to pixel values right away. Both of these steps are fairly straightforward using the Canvas API.

For an example implementation of this technique see the [image classification example](image_classification.exs).

### User audio

The points made about images above are relevant to user-given audio as well. In fact, decoding audio files on the server requires ffmpeg to be installed system-wide. However, we can do all preprocessing on the client and send raw PCM data with a single channel to the server.

For an example implementation of this technique see the [speech-to-text example](speech_to_text.exs).

If you are interested in real-time streaming, look at the [Membrane Framework](https://github.com/membraneframework/membrane_core).
