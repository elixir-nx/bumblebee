Mix.install([
  {:bumblebee, path: Path.expand("..", __DIR__)},
  {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
  {:exla, github: "elixir-nx/nx", sparse: "exla"},
  {:stb_image, "~> 0.5.0"}
])

Nx.default_backend(EXLA.Backend)

auth_token = System.fetch_env!("HF_TOKEN")

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

{:ok, clip_model, clip_params, clip_spec} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "text_encoder"},
    architecture: :base
  )

{:ok, vae_model, vae_params, vae_spec} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "vae"},
    architecture: :decoder,
    params_filename: "diffusion_pytorch_model.bin"
  )

{:ok, unet_model, unet_params, unet_spec} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "unet"},
    params_filename: "diffusion_pytorch_model.bin"
  )

{:ok, scheduler} =
  Bumblebee.load_scheduler(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "scheduler"}
  )

prompt = "numbat in forest, detailed, digital art"
num_images_per_prompt = 2

entries =
  Bumblebee.Diffusion.StableDiffusion.text_to_image(
    {clip_model, clip_params, clip_spec},
    {vae_model, vae_params, vae_spec},
    {unet_model, unet_params, unet_spec},
    tokenizer,
    scheduler,
    prompt,
    num_steps: 20,
    num_images_per_prompt: num_images_per_prompt
  )

for {entry, idx} <- Enum.with_index(entries) do
  entry.image
  |> StbImage.from_nx()
  |> StbImage.write_file!("tmp/sample_#{idx}.png")
end
