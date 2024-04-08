defmodule Bumblebee.Diffusion.StableDiffusionControlNetTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  describe "text_to_image/6" do
    test "generates image for a text prompt with controlnet" do
      # Since we don't assert on the result in this case, we use
      # a tiny random checkpoint. This test is basically to verify
      # the whole generation computation end-to-end

      repository_id = "bumblebee-testing/tiny-stable-diffusion"

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})
      {:ok, clip} = Bumblebee.load_model({:hf, repository_id, subdir: "text_encoder"})

      {:ok, unet} = Bumblebee.load_model({:hf, repository_id, subdir: "unet"})

      {:ok, controlnet} = Bumblebee.load_model({:hf, "bumblebee-testing/tiny-controlnet"})

      {:ok, vae} =
        Bumblebee.load_model({:hf, repository_id, subdir: "vae"}, architecture: :decoder)

      {:ok, scheduler} = Bumblebee.load_scheduler({:hf, repository_id, subdir: "scheduler"})

      {:ok, featurizer} =
        Bumblebee.load_featurizer({:hf, repository_id, subdir: "feature_extractor"})

      {:ok, safety_checker} = Bumblebee.load_model({:hf, repository_id, subdir: "safety_checker"})

      conditioning_size =
        controlnet.spec.sample_size *
          2 ** (length(controlnet.spec.conditioning_embedding_hidden_sizes) - 1)

      serving =
        Bumblebee.Diffusion.StableDiffusionControlNet.text_to_image(
          clip,
          unet,
          vae,
          controlnet,
          tokenizer,
          scheduler,
          num_steps: 3,
          safety_checker: safety_checker,
          safety_checker_featurizer: featurizer
        )

      prompt = "numbat in forest, detailed, digital art"

      conditioning =
        Nx.broadcast(Nx.tensor(50, type: :u8), {conditioning_size, conditioning_size, 3})

      assert %{
               results: [%{image: %Nx.Tensor{}, is_safe: _boolean}]
             } =
               Nx.Serving.run(serving, %{
                 prompt: prompt,
                 conditioning: conditioning
               })

      # Without safety checker

      serving =
        Bumblebee.Diffusion.StableDiffusionControlNet.text_to_image(
          clip,
          unet,
          vae,
          controlnet,
          tokenizer,
          scheduler,
          num_steps: 3
        )

      prompt = "numbat in forest, detailed, digital art"

      assert %{results: [%{image: %Nx.Tensor{}}]} =
               Nx.Serving.run(serving, %{
                 prompt: prompt,
                 conditioning: conditioning
               })

      # With compilation

      serving =
        Bumblebee.Diffusion.StableDiffusionControlNet.text_to_image(
          clip,
          unet,
          vae,
          controlnet,
          tokenizer,
          scheduler,
          num_steps: 3,
          safety_checker: safety_checker,
          safety_checker_featurizer: featurizer,
          compile: [batch_size: 1, sequence_length: 60],
          defn_options: [compiler: EXLA]
        )

      prompt = "numbat in forest, detailed, digital art"

      assert %{
               results: [%{image: %Nx.Tensor{}, is_safe: _boolean}]
             } =
               Nx.Serving.run(serving, %{
                 prompt: prompt,
                 conditioning: conditioning
               })
    end
  end
end
