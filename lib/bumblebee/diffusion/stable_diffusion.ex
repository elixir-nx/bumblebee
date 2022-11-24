defmodule Bumblebee.Diffusion.StableDiffusion do
  @moduledoc """
  High-level tasks based on Stable Diffusion.
  """

  import Nx.Defn

  alias Bumblebee.Shared

  @type text_to_image_input :: String.t()
  @type text_to_image_output :: %{results: list(text_to_image_result())}
  @type text_to_image_result :: %{:image => Nx.Tensor.t(), optional(:is_safe) => boolean()}

  @doc ~S"""
  Build serving for prompt-driven image generation.

  The serving accepts `t:text_to_image_input/0` and returns `t:text_to_image_output/0`.
  A list of inputs is also supported.

  You can specify `:safety_checker` model to automatically detect
  when a generated image is offensive or harmful and filter it out.

  ## Options

    * `:safety_checker` - the safety checker model info map. When a
      safety checker is used, each output entry has an additional
      `:is_safe` property and unsafe images are automatically zeroed.
      Make sure to also set `:safety_checker_featurizer`

    * `:safety_checker_featurizer` - the featurizer to use to preprocess
      the safety checker input images

    * `:num_steps` - the number of denoising steps. More denoising
      steps usually lead to higher image quality at the expense of
      slower inference. Defaults to `50`

    * `:num_images_per_prompt` - the number of images to generate for
      each prompt. Defaults to `1`

    * `:guidance_scale` - the scale used for classifier-free diffusion
      guidance. Higher guidance scale makes the generated images more
      closely reflect the text prompt. This parameter corresponds to
      $\omega$ in Equation (2) of the [Imagen paper](https://arxiv.org/pdf/2205.11487.pdf).
      Defaults to `7.5`

    * `:seed` - a seed for the random number generator. Defaults to `0`

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

        * `:sequence_length` - the maximum input sequence length. Input
          sequences are always padded/truncated to match that length

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  ## Examples

      auth_token = System.fetch_env!("HF_TOKEN")

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

      {:ok, clip} =
        Bumblebee.load_model(
          {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "text_encoder"}
        )

      {:ok, unet} =
        Bumblebee.load_model(
          {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "unet"},
          params_filename: "diffusion_pytorch_model.bin"
        )

      {:ok, vae} =
        Bumblebee.load_model(
          {:hf, "CompVis/stable-diffusion-v1-4", auth_token: auth_token, subdir: "vae"},
          architecture: :decoder,
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

      serving =
        Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, unet, vae, tokenizer, scheduler,
          num_steps: 20,
          num_images_per_prompt: 2,
          safety_checker: safety_checker,
          safety_checker_featurizer: featurizer,
          compile: [batch_size: 1, sequence_length: 60],
          defn_options: [compiler: EXLA]
        )

      prompt = "numbat in forest, detailed, digital art"
      Nx.Serving.run(serving, prompt)
      #=> %{
      #=>   results: [
      #=>     %{
      #=>       image: #Nx.Tensor<
      #=>         u8[512][512][3]
      #=>         ...
      #=>       >,
      #=>       is_safe: true
      #=>     },
      #=>     %{
      #=>       image: #Nx.Tensor<
      #=>         u8[512][512][3]
      #=>         ...
      #=>       >,
      #=>       is_safe: true
      #=>     }
      #=>   ]
      #=> }

  """
  @spec text_to_image(
          Bumblebee.model_info(),
          Bumblebee.model_info(),
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          Bumblebee.Scheduler.t(),
          keyword()
        ) :: Nx.Serving.t()
  def text_to_image(encoder, unet, vae, tokenizer, scheduler, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :safety_checker,
        :safety_checker_featurizer,
        :compile,
        num_steps: 50,
        num_images_per_prompt: 1,
        guidance_scale: 7.5,
        seed: 0,
        defn_options: []
      ])

    safety_checker = opts[:safety_checker]
    safety_checker_featurizer = opts[:safety_checker_featurizer]
    num_steps = opts[:num_steps]
    num_images_per_prompt = opts[:num_images_per_prompt]
    compile = opts[:compile]
    defn_options = opts[:defn_options]

    if safety_checker != nil and safety_checker_featurizer == nil do
      raise ArgumentError, "got :safety_checker but no :safety_checker_featurizer was specified"
    end

    safety_checker? = safety_checker != nil

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    {_, encoder_predict} = Axon.build(encoder.model)
    {_, vae_predict} = Axon.build(vae.model)
    {_, unet_predict} = Axon.build(unet.model)

    scheduler_init = fn latents_shape ->
      Bumblebee.scheduler_init(scheduler, num_steps, latents_shape)
    end

    scheduler_step = &Bumblebee.scheduler_step(scheduler, &1, &2, &3)

    images_fun =
      &text_to_image_impl(
        encoder_predict,
        &1,
        unet_predict,
        &2,
        vae_predict,
        &3,
        scheduler_init,
        scheduler_step,
        &4,
        num_images_per_prompt: opts[:num_images_per_prompt],
        latents_sample_size: unet.spec.sample_size,
        latents_channels: unet.spec.in_channels,
        seed: opts[:seed],
        guidance_scale: opts[:guidance_scale]
      )

    safety_checker_fun =
      if safety_checker do
        {_, predict_fun} = Axon.build(safety_checker.model)
        predict_fun
      end

    # Note that all of these are copied when using serving as a process
    init_args = [
      {images_fun, safety_checker_fun},
      {encoder.params, encoder.spec},
      unet.params,
      vae.params,
      {safety_checker?, safety_checker[:spec], safety_checker[:params]},
      safety_checker_featurizer,
      {compile != nil, batch_size, sequence_length},
      num_images_per_prompt,
      defn_options
    ]

    Nx.Serving.new(fn -> apply(&init/9, init_args) end, batch_size: batch_size)
    |> Nx.Serving.client_preprocessing(&client_preprocessing(&1, tokenizer, sequence_length))
    |> Nx.Serving.client_postprocessing(
      &client_postprocessing(&1, &2, &3, num_images_per_prompt, safety_checker)
    )
  end

  defp init(
         {images_fun, safety_checker_fun},
         {encoder_params, encoder_spec},
         unet_params,
         vae_params,
         {safety_checker?, safety_checker_spec, safety_checker_params},
         safety_checker_featurizer,
         {compile?, batch_size, sequence_length},
         num_images_per_prompt,
         defn_options
       ) do
    images_fun =
      if compile? do
        text_inputs_template = %{
          "input_ids" =>
            Shared.input_template(encoder_spec, "input_ids", [
              batch_size,
              sequence_length
            ])
        }

        input_template = %{
          "unconditional" => text_inputs_template,
          "conditional" => text_inputs_template
        }

        template_args =
          Shared.templates([encoder_params, unet_params, vae_params, input_template])

        Nx.Defn.compile(images_fun, template_args, defn_options)
      else
        Nx.Defn.jit(images_fun, defn_options)
      end

    safety_checker_fun =
      safety_checker_fun &&
        if compile? do
          input_template = %{
            "pixel_values" =>
              Shared.input_template(safety_checker_spec, "pixel_values", [
                batch_size * num_images_per_prompt
              ])
          }

          template_args = [Nx.to_template(safety_checker_params), input_template]
          Nx.Defn.compile(safety_checker_fun, template_args, defn_options)
        else
          Nx.Defn.jit(safety_checker_fun, defn_options)
        end

    &Shared.with_optional_padding(&1, batch_size, fn inputs ->
      images = images_fun.(encoder_params, unet_params, vae_params, inputs)

      output =
        if safety_checker? do
          inputs = Bumblebee.apply_featurizer(safety_checker_featurizer, images)
          outputs = safety_checker_fun.(safety_checker_params, inputs)
          %{images: images, is_unsafe: outputs.is_unsafe}
        else
          %{images: images}
        end

      Bumblebee.Utils.Nx.composite_unflatten_batch(output, inputs.size)
    end)
  end

  defp client_preprocessing(input, tokenizer, sequence_length) do
    {prompts, multi?} = Shared.validate_serving_input!(input, &is_binary/1, "a string")
    num_prompts = length(prompts)

    conditional =
      Bumblebee.apply_tokenizer(tokenizer, prompts,
        length: sequence_length,
        return_token_type_ids: false,
        return_attention_mask: false
      )

    unconditional =
      Bumblebee.apply_tokenizer(tokenizer, List.duplicate("", num_prompts),
        length: Nx.axis_size(conditional["input_ids"], 1),
        return_attention_mask: false,
        return_token_type_ids: false
      )

    inputs = %{"unconditional" => unconditional, "conditional" => conditional}

    {Nx.Batch.concatenate([inputs]), {num_prompts, multi?}}
  end

  defp client_postprocessing(
         outputs,
         _metadata,
         {num_prompts, multi?},
         num_images_per_prompt,
         safety_checker?
       ) do
    for idx <- 0..(num_prompts - 1) do
      results =
        for result_idx <- 0..(num_images_per_prompt - 1) do
          image = outputs.images[idx][result_idx]

          if safety_checker? do
            if Nx.to_number(outputs.is_unsafe[idx][result_idx]) == 1 do
              %{image: zeroed(image), is_safe: false}
            else
              %{image: image, is_safe: true}
            end
          else
            %{image: image}
          end
        end

      %{results: results}
    end
    |> Shared.normalize_output(multi?)
  end

  defp zeroed(tensor) do
    0
    |> Nx.tensor(type: Nx.type(tensor))
    |> Nx.broadcast(Nx.shape(tensor))
  end

  defnp text_to_image_impl(
          encoder_predict,
          encoder_params,
          unet_predict,
          unet_params,
          vae_predict,
          vae_params,
          scheduler_init,
          scheduler_step,
          inputs,
          opts \\ []
        ) do
    num_images_per_prompt = opts[:num_images_per_prompt]
    latents_sample_size = opts[:latents_sample_size]
    latents_in_channels = opts[:latents_channels]
    seed = opts[:seed]
    guidance_scale = opts[:guidance_scale]

    inputs =
      Bumblebee.Utils.Nx.composite_concatenate(inputs["unconditional"], inputs["conditional"])

    %{hidden_state: text_embeddings} = encoder_predict.(encoder_params, inputs)

    {twice_batch_size, seq_length, hidden_size} = Nx.shape(text_embeddings)
    batch_size = div(twice_batch_size, 2)

    text_embeddings =
      text_embeddings
      |> Nx.new_axis(1)
      |> Nx.tile([1, num_images_per_prompt, 1, 1])
      |> Nx.reshape({:auto, seq_length, hidden_size})

    latents_shape =
      {batch_size * num_images_per_prompt, latents_sample_size, latents_sample_size,
       latents_in_channels}

    {scheduler_state, timesteps} = scheduler_init.(latents_shape)

    key = Nx.Random.key(seed)
    {latents, _key} = Nx.Random.normal(key, shape: latents_shape)

    {_, latents, _, _} =
      while {scheduler_state, latents, text_embeddings, unet_params}, timestep <- timesteps do
        unet_inputs = %{
          "sample" => Nx.concatenate([latents, latents]),
          "timestep" => timestep,
          "encoder_hidden_state" => text_embeddings
        }

        %{sample: noise_pred} = unet_predict.(unet_params, unet_inputs)

        {noise_pred_unconditional, noise_pred_text} = split_in_half(noise_pred)

        noise_pred =
          noise_pred_unconditional + guidance_scale * (noise_pred_text - noise_pred_unconditional)

        {scheduler_state, latents} = scheduler_step.(scheduler_state, latents, noise_pred)

        {scheduler_state, latents, text_embeddings, unet_params}
      end

    latents = latents * (1 / 0.18215)

    %{sample: images} = vae_predict.(vae_params, latents)

    NxImage.from_continuous(images, -1, 1)
  end

  defnp split_in_half(tensor) do
    batch_size = Nx.axis_size(tensor, 0)
    half_size = div(batch_size, 2)
    {tensor[0..(half_size - 1)//1], tensor[half_size..-1//1]}
  end
end
