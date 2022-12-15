defmodule Bumblebee.Diffusion.StableDiffusionTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "text_to_image/6" do
      repository_id = "CompVis/stable-diffusion-v1-4"

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

      {:ok, clip} = Bumblebee.load_model({:hf, repository_id, subdir: "text_encoder"})

      {:ok, unet} =
        Bumblebee.load_model({:hf, repository_id, subdir: "unet"},
          params_filename: "diffusion_pytorch_model.bin"
        )

      {:ok, vae} =
        Bumblebee.load_model({:hf, repository_id, subdir: "vae"},
          architecture: :decoder,
          params_filename: "diffusion_pytorch_model.bin"
        )

      {:ok, scheduler} = Bumblebee.load_scheduler({:hf, repository_id, subdir: "scheduler"})

      {:ok, featurizer} =
        Bumblebee.load_featurizer({:hf, repository_id, subdir: "feature_extractor"})

      {:ok, safety_checker} = Bumblebee.load_model({:hf, repository_id, subdir: "safety_checker"})

      serving =
        Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, unet, vae, tokenizer, scheduler,
          num_steps: 2,
          safety_checker: safety_checker,
          safety_checker_featurizer: featurizer,
          defn_options: [compiler: EXLA]
        )

      prompt = "numbat in forest, detailed, digital art"

      assert %{
               results: [%{image: %Nx.Tensor{}, is_safe: _boolean}]
             } = Nx.Serving.run(serving, prompt)
    end
  end
end
