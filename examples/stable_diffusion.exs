Mix.install([
  {:bumblebee, path: Path.expand("..", __DIR__)},
  {:nx, "~> 0.4.0"},
  {:exla, "~> 0.4.0"},
  {:stb_image, "~> 0.5.0"}
])

Nx.default_backend(EXLA.Backend)

auth_token = System.fetch_env!("HF_TOKEN")

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

{:ok, clip} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "text_encoder"}
  )

{:ok, vae} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "vae"},
    architecture: :decoder,
    params_filename: "diffusion_pytorch_model.bin"
  )

{:ok, unet} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "unet"},
    params_filename: "diffusion_pytorch_model.bin"
  )

{:ok, scheduler} =
  Bumblebee.load_scheduler(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "scheduler"}
  )

{:ok, featurizer} =
  Bumblebee.load_featurizer(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "feature_extractor"}
  )

{:ok, safety_checker} =
  Bumblebee.load_model(
    {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "safety_checker"}
  )

prompt = "numbat in forest, detailed, digital art"
num_images_per_prompt = 2

entries =
  Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, vae, unet, tokenizer, scheduler, prompt,
    num_steps: 20,
    num_images_per_prompt: num_images_per_prompt,
    safety_checker: safety_checker,
    safety_checker_featurizer: featurizer,
    defn_options: [compiler: EXLA]
  )

for {entry, idx} <- Enum.with_index(entries) do
  entry.image
  |> StbImage.from_nx()
  |> StbImage.write_file!("tmp/sample_#{idx}.png")
end
