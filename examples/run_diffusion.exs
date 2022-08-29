defmodule Image do
  import Nx.Defn

  defn normalize(image) do
    mn = Nx.reduce_min(image)
    mx = Nx.reduce_max(image)

    mx = mx - mn

    image = ((image - mn) / mx) * 255
    Nx.as_type(image, {:u, 8})
  end
end

Nx.Defn.global_default_options(compiler: EXLA)
Nx.default_backend(EXLA.Backend)

# Parameters
batch_size = 1
in_channels = 4
height = 512
width = 512
num_inference_steps = 10
max_length = 77

# Models
IO.puts("Loading CLIP")
{:ok, clip, clip_params, clip_config} =
  Bumblebee.load_model({:local, "test/models/stable-diffusion-v1-1/text_encoder"},
    module: Bumblebee.Text.ClipText,
    architecture: :base
  )

# You must convert the slow tokenizer to fast
IO.puts("Loading tokenizer")
{:ok, tokenizer} = Bumblebee.load_tokenizer({:local, "test/models/stable-diffusion-v1-1/tokenizer"}, module: Bumblebee.Text.ClipTokenizer)

IO.write("Loading VAE")
{:ok, vae, vae_params, vae_config} =
  Bumblebee.load_model({:local, "test/models/stable-diffusion-v1-1/vae"},
    module: Bumblebee.Diffusion.AutoencoderKl,
    architecture: :decoder
  )

IO.puts("Loading UNet")
{:ok, unet, unet_params, unet_config} =
  Bumblebee.load_model({:local, "test/models/stable-diffusion-v1-1/unet"},
    module: Bumblebee.Diffusion.UNet2DCondition,
    architecture: :base
  )

IO.puts("Loading scheduler")
step = Bumblebee.Diffusion.Schedules.ddim(num_inference_steps)

# # Prompt
prompt = "an astronaut riding a horse"
inputs = Bumblebee.apply_tokenizer(tokenizer, [prompt])

IO.puts("Embedding text...")
text_embeddings = Axon.predict(clip, clip_params, inputs, compiler: EXLA)
text_embeddings = text_embeddings.last_hidden_state

IO.puts("Generating latents...")
latents_shape = {batch_size, in_channels, div(height, 8), div(width, 8)}
latents = Nx.random_normal(latents_shape)

timesteps =
  Nx.iota({num_inference_steps})
  |> Nx.multiply(Nx.quotient(1000, num_inference_steps))
  |> Nx.reverse()
  |> Nx.add(1)

latents =
  for {timestep, i} <- Enum.with_index(Nx.to_flat_list(timesteps)), reduce: latents do
    latents ->
      IO.puts("Diffusion step #{i}, timestep #{timestep}")
      unet_inputs = %{
        "sample" => latents,
        "timestep" => Nx.tensor(timestep),
        "encoder_hidden_states" => text_embeddings
      }
      noise_pred = Axon.predict(unet, unet_params, unet_inputs, compiler: EXLA)
      step.(noise_pred, timestep, latents)
  end

latents = Nx.multiply(Nx.divide(1, 0.18215), latents)
image = Axon.predict(vae, vae_params, latents, compiler: EXLA)
image = Nx.divide(image, 2) |> Nx.add(0.5) |> Nx.clip(0, 1)
image = Nx.transpose(image, axes: [0, 2, 3, 1])

image
|> Nx.squeeze()
|> Image.normalize()
|> StbImage.from_nx()
|> StbImage.write_file("sample.png")
