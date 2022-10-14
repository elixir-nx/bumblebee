Mix.install([
  {:bumblebee, path: Path.expand("..", __DIR__)},
  {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
  {:exla, github: "elixir-nx/nx", sparse: "exla"},
  {:stb_image, "~> 0.5.0"}
])

Nx.default_backend(EXLA.Backend)

# Parameters

prompt = "numbat in forest, detailed, digital art"
batch_size = 2
num_steps = 20
guidance_scale = 7.5
seed = 0

auth_token = System.fetch_env!("HF_TOKEN")

# Models

IO.puts("Loading CLIP")

{:ok, clip, clip_params, _clip_spec} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "text_encoder"},
    architecture: :base
  )

IO.puts("Loading tokenizer")

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

IO.puts("Loading VAE")

{:ok, vae, vae_params, _vae_spec} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "vae"},
    architecture: :decoder,
    params_filename: "diffusion_pytorch_model.bin"
  )

IO.puts("Loading UNet")

{:ok, unet, unet_params, unet_spec} =
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

prompts = List.duplicate("", batch_size) ++ List.duplicate(prompt, batch_size)
inputs = Bumblebee.apply_tokenizer(tokenizer, prompts)

latents_shape = {batch_size, unet_spec.in_channels, unet_spec.sample_size, unet_spec.sample_size}

IO.puts("Embedding text")
%{hidden_state: text_embeddings} = Axon.predict(clip, clip_params, inputs)

IO.puts("Generating latents")
key = Nx.Random.key(seed)
{latents, _key} = Nx.Random.normal(key, shape: latents_shape)

{scheduler_state, timesteps} = Bumblebee.scheduler_init(scheduler, num_steps, latents_shape)

timesteps = Nx.to_flat_list(timesteps)

{_, latents} =
  for {timestep, i} <- Enum.with_index(timesteps), reduce: {scheduler_state, latents} do
    {scheduler_state, latents} ->
      IO.puts("Iteration #{i}, timestep #{timestep}")

      unet_inputs = %{
        "sample" => Nx.concatenate([latents, latents]),
        "timestep" => Nx.tensor(timestep),
        "encoder_hidden_state" => text_embeddings
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
%{sample: images} = Axon.predict(vae, vae_params, latents)

images =
  images
  |> Bumblebee.Utils.Image.from_continuous(-1, 1)
  |> Nx.transpose(axes: [0, 2, 3, 1])

for i <- 0..(batch_size - 1) do
  images[i]
  |> StbImage.from_nx()
  |> StbImage.write_file!("tmp/sample_#{i}.png")
end
