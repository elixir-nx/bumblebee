Nx.default_backend(EXLA.Backend)

# Parameters

batch_size = 1
in_channels = 4
height = 512
width = 512
num_inference_steps = 10
guidance_scale = 7.5
latents_shape = {batch_size, in_channels, div(height, 8), div(width, 8)}

# Models

IO.puts("Loading CLIP")

{:ok, clip, clip_params, _clip_config} =
  Bumblebee.load_model({:local, "tmp/models/stable-diffusion-v1-4/text_encoder"},
    module: Bumblebee.Text.ClipText,
    architecture: :base
  )

IO.puts("Loading tokenizer")

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

IO.puts("Loading VAE")

{:ok, vae, vae_params, _vae_config} =
  Bumblebee.load_model({:local, "tmp/models/stable-diffusion-v1-4/vae"},
    module: Bumblebee.Diffusion.AutoencoderKl,
    architecture: :decoder
  )

IO.puts("Loading UNet")

{:ok, unet, unet_params, _unet_config} =
  Bumblebee.load_model({:local, "tmp/models/stable-diffusion-v1-4/unet"},
    module: Bumblebee.Diffusion.UNet2DCondition,
    architecture: :base
  )

IO.puts("Loading scheduler")

# TODO: load the scheduler and then init, perhaps a protocol (so that changing scheduler
# doesn't change the rest of the code)
scheduler =
  Bumblebee.Diffusion.Schedule.Pndm.new(num_inference_steps, latents_shape,
    beta_schedule: :scaled_linear,
    beta_start: 0.00085,
    beta_end: 0.012,
    num_train_timesteps: 1000,
    offset: 1
  )

# Inference

prompt = "an astronaut riding a horse"
inputs = Bumblebee.apply_tokenizer(tokenizer, ["", prompt])

IO.puts("Embedding text")
%{last_hidden_state: text_embeddings} = Axon.predict(clip, clip_params, inputs, compiler: EXLA)

IO.puts("Generating latents")
latents = Nx.random_normal(latents_shape)

timesteps = Bumblebee.Diffusion.Schedule.Pndm.timesteps(scheduler)

{_, latents} =
  for {timestep, i} <- Enum.with_index(timesteps), reduce: {scheduler, latents} do
    {scheduler, latents} ->
      IO.puts("Diffusion step #{i}, timestep #{timestep}")

      unet_inputs = %{
        "sample" => Nx.concatenate([latents, latents]),
        "timestep" => Nx.tensor(timestep),
        "encoder_hidden_states" => text_embeddings
      }

      noise_pred = Axon.predict(unet, unet_params, unet_inputs, compiler: EXLA)

      noise_pred_unconditional = noise_pred[0..(batch_size - 1)//1]
      noise_pred_text = noise_pred[batch_size..-1//1]

      noise_pred =
        Nx.add(
          noise_pred_unconditional,
          Nx.multiply(guidance_scale, Nx.subtract(noise_pred_text, noise_pred_unconditional))
        )

      Bumblebee.Diffusion.Schedule.Pndm.step(scheduler, noise_pred, timestep, latents)
  end

latents = Nx.multiply(Nx.divide(1, 0.18215), latents)
images = Axon.predict(vae, vae_params, latents, compiler: EXLA)

images =
  images
  |> Bumblebee.Utils.Image.from_continuous(-1, 1)
  |> Nx.transpose(axes: [0, 2, 3, 1])

images[0]
|> StbImage.from_nx()
|> StbImage.write_file!("tmp/sample.png")
