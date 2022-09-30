Nx.default_backend(EXLA.Backend)

# Parameters

batch_size = 1
in_channels = 4
height = 512
width = 512
num_inference_steps = 10
guidance_scale = 7.5
latents_shape = {batch_size, in_channels, div(height, 8), div(width, 8)}

auth_token = System.fetch_env!("HF_TOKEN")

# Models

IO.puts("Loading CLIP")

{:ok, clip, clip_params, _clip_config} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "text_encoder"},
    architecture: :base
  )

IO.puts("Loading tokenizer")

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

IO.puts("Loading VAE")

{:ok, vae, vae_params, _vae_config} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "vae"},
    architecture: :decoder,
    params_filename: "diffusion_pytorch_model.bin"
  )

IO.puts("Loading UNet")

{:ok, unet, unet_params, _unet_config} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "unet"},
    params_filename: "diffusion_pytorch_model.bin"
  )

IO.puts("Loading scheduler")

{:ok, scheduler} =
  Bumblebee.load_scheduler(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "scheduler"}
  )

# Inference

prompt = "a photograph of an astronaut riding a horse"
inputs = Bumblebee.apply_tokenizer(tokenizer, ["", prompt])

IO.puts("Embedding text")
%{last_hidden_state: text_embeddings} = Axon.predict(clip, clip_params, inputs, compiler: EXLA)

IO.puts("Generating latents")
latents = Nx.random_normal(latents_shape)

{scheduler_state, timesteps} =
  Bumblebee.scheduler_init(scheduler, num_inference_steps, Nx.shape(latents))

timesteps = Nx.to_flat_list(timesteps)

{_, latents} =
  for {timestep, i} <- Enum.with_index(timesteps), reduce: {scheduler_state, latents} do
    {scheduler_state, latents} ->
      IO.puts("Diffusion step #{i}, timestep #{timestep}")

      unet_inputs = %{
        "sample" => Nx.concatenate([latents, latents]),
        "timestep" => Nx.tensor(timestep),
        "encoder_last_hidden_state" => text_embeddings
      }

      %{sample: noise_pred} = Axon.predict(unet, unet_params, unet_inputs, compiler: EXLA)

      noise_pred_unconditional = noise_pred[0..(batch_size - 1)//1]
      noise_pred_text = noise_pred[batch_size..-1//1]

      noise_pred =
        Nx.add(
          noise_pred_unconditional,
          Nx.multiply(guidance_scale, Nx.subtract(noise_pred_text, noise_pred_unconditional))
        )

      Bumblebee.scheduler_step(scheduler, scheduler_state, latents, noise_pred)
  end

latents = Nx.multiply(Nx.divide(1, 0.18215), latents)
%{sample: images} = Axon.predict(vae, vae_params, latents, compiler: EXLA)

images =
  images
  |> Bumblebee.Utils.Image.from_continuous(-1, 1)
  |> Nx.transpose(axes: [0, 2, 3, 1])

images[0]
|> StbImage.from_nx()
|> StbImage.write_file!("tmp/sample.png")
