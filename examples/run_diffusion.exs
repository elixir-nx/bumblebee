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

scheduler_config =
  Bumblebee.Diffusion.Schedules.Pndm.new(
    beta_schedule: :scaled_linear,
    beta_start: 0.00085,
    beta_end: 0.012,
    skip_prk_steps: true,
    num_train_timesteps: 1000
  )

# Inference

prompt = "a photograph of an astronaut riding a horse"
inputs = Bumblebee.apply_tokenizer(tokenizer, ["", prompt])

IO.puts("Embedding text")
%{last_hidden_state: text_embeddings} = Axon.predict(clip, clip_params, inputs, compiler: EXLA)

IO.puts("Generating latents")
latents = Nx.random_normal(latents_shape)

{schedule, timesteps} =
  Bumblebee.Diffusion.Schedules.Pndm.init(scheduler_config, num_inference_steps, Nx.shape(latents),
    offset: 1
  )

{_, latents} =
  for {timestep, i} <- Enum.with_index(timesteps), reduce: {schedule, latents} do
    {schedule, latents} ->
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

      Bumblebee.Diffusion.Schedules.Pndm.step(schedule, latents, noise_pred)
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
