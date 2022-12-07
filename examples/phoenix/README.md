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

  2. Whenever we want to make a prediction for a user input, we use `Nx.Serving.batched_run/2` with said input, usually in a `Task` to avoid blocking other interactions. Most importantly, `Nx.Serving` automatically batches requests from concurrent users and distributes the results back to them transparently.

## Tips

### Deployment considerations

The examples in this directory download the model data directly from HuggingFace the first time the application starts. In practice you'd version those files alongside your repository (perhaps using [Git LFS](https://git-lfs.github.com/)) or use an object storage service. Then, you'd fetch those files into a local directory as part of deployment.

Once that is done, you can change the calls from `Bumblebee.load_xyz({:hf, "microsoft/resnet"})` to `Bumblebee.load_xyz({:local, "/path/to/model"})` and so on.

### User images

When working with user-given images, the most trivial approach would be to just upload an image as is, in a format like PNG or JPEG. However, this approach has two downsides:

  1. In most cases full-resolution images are not necessary, because Neural Networks work on much smaller inputs. This means that a lot of data is unnecessarily uploaded over the network and that the server needs to do more work downsizing a potentially large image.

  2. Decoding an image format like PNG or JPEG requires an additional package and again, puts more work on the server.

Both of these downsides can be avoided by moving all the work to the client. Specifically, when the user selects an image, we can resize it to a much smaller version and decode to pixel values right away. Both of these steps are fairly straightforward using the Canvas API.

For an example implementation of this technique see the [image classification example](image_classification.exs).
