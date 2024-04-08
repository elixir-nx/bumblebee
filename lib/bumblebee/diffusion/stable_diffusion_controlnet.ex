defmodule Bumblebee.Diffusion.StableDiffusionControlNet do
  @moduledoc """
  High-level tasks based on Stable Diffusion with ControlNet.
  """

  import Nx.Defn

  alias Bumblebee.Utils
  alias Bumblebee.Shared

  @type text_to_image_input ::
          String.t()
          | %{
              :prompt => String.t(),
              :conditioning => Nx.Tensor.t(),
              optional(:conditioning_scale) => integer(),
              optional(:negative_prompt) => String.t(),
              optional(:seed) => integer()
            }
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

    * `:preallocate_params` - when `true`, explicitly allocates params
      on the device configured by `:defn_options`. You may want to set
      this option when using partitioned serving, to allocate params
      on each of the devices. When using this option, you should first
      load the parameters into the host. This can be done by passing
      `backend: {EXLA.Backend, client: :host}` to `load_model/1` and friends.
      Defaults to `false`

  ## Examples

      repository_id = "CompVis/stable-diffusion-v1-4"

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})
      {:ok, clip} = Bumblebee.load_model({:hf, repository_id, subdir: "text_encoder"})
      {:ok, unet} = Bumblebee.load_model({:hf, repository_id, subdir: "unet"})
      {:ok, controlnet} = Bumblebee.load_model({:hf, "lllyasviel/sd-controlnet-scribble"})
      {:ok, vae} = Bumblebee.load_model({:hf, repository_id, subdir: "vae"}, architecture: :decoder)
      {:ok, scheduler} = Bumblebee.load_scheduler({:hf, repository_id, subdir: "scheduler"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, repository_id, subdir: "feature_extractor"})
      {:ok, safety_checker} = Bumblebee.load_model({:hf, repository_id, subdir: "safety_checker"})

      serving =
        Bumblebee.Diffusion.StableDiffusionControlNet.text_to_image(
          clip,
          unet,
          vae,
          controlnet,
          tokenizer,
          scheduler,
          num_steps: 20,
          num_images_per_prompt: 2,
          safety_checker: safety_checker,
          safety_checker_featurizer: featurizer,
          compile: [batch_size: 1, sequence_length: 60],
          defn_options: [compiler: EXLA]
        )

      prompt = "numbat in forest, detailed, digital art"

      conditioning =
        Nx.tensor(
          [for(_ <- 1..8, do: [255]) ++ for(_ <- 1..24, do: [0])],
          type: :u8
        )
        |> Nx.tile([256, 8, 3])
        |> Nx.pad(0, [{192, 64, 0}, {192, 64, 0}, {0, 0, 0}])
        |> Nx.transpose(axes: [1, 0, 2])

      Nx.Serving.run(serving, %{prompt: prompt, conditioning: conditioning})
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
          Bumblebee.model_info(),
          Bumblebee.Tokenizer.t(),
          Bumblebee.Scheduler.t(),
          keyword()
        ) :: Nx.Serving.t()
  def text_to_image(encoder, unet, vae, controlnet, tokenizer, scheduler, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :safety_checker,
        :safety_checker_featurizer,
        :compile,
        num_steps: 50,
        num_images_per_prompt: 1,
        guidance_scale: 7.5,
        defn_options: [],
        preallocate_params: false
      ])

    safety_checker = opts[:safety_checker]
    safety_checker_featurizer = opts[:safety_checker_featurizer]
    num_steps = opts[:num_steps]
    num_images_per_prompt = opts[:num_images_per_prompt]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    if safety_checker != nil and safety_checker_featurizer == nil do
      raise ArgumentError, "got :safety_checker but no :safety_checker_featurizer was specified"
    end

    safety_checker? = safety_checker != nil

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size, :sequence_length])
        |> Shared.require_options!([:batch_size, :sequence_length])
      end

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    conditioning_size =
      controlnet.spec.sample_size *
        2 ** (length(controlnet.spec.conditioning_embedding_hidden_sizes) - 1)

    tokenizer =
      Bumblebee.configure(tokenizer,
        length: sequence_length,
        return_token_type_ids: false,
        return_attention_mask: false
      )

    {_, encoder_predict} = Axon.build(encoder.model)
    {_, vae_predict} = Axon.build(vae.model)
    {_, unet_predict} = Axon.build(unet.model)
    {_, controlnet_predict} = Axon.build(controlnet.model)

    scheduler_init = &Bumblebee.scheduler_init(scheduler, num_steps, &1, &2)
    scheduler_step = &Bumblebee.scheduler_step(scheduler, &1, &2, &3)

    image_fun =
      &text_to_image_impl(
        encoder_predict,
        &1,
        unet_predict,
        &2,
        vae_predict,
        &3,
        controlnet_predict,
        &4,
        scheduler_init,
        scheduler_step,
        &5,
        num_images_per_prompt: opts[:num_images_per_prompt],
        latents_sample_size: unet.spec.sample_size,
        latents_channels: unet.spec.in_channels,
        guidance_scale: opts[:guidance_scale]
      )

    safety_checker_fun =
      if safety_checker do
        {_, predict_fun} = Axon.build(safety_checker.model)
        predict_fun
      end

    # Note that all of these are copied when using serving as a process
    init_args = [
      {image_fun, safety_checker_fun},
      encoder.params,
      unet.params,
      vae.params,
      controlnet.params,
      {safety_checker?, safety_checker[:spec], safety_checker[:params]},
      safety_checker_featurizer,
      {compile != nil, batch_size, sequence_length, conditioning_size},
      num_images_per_prompt,
      preallocate_params
    ]

    Nx.Serving.new(
      fn defn_options -> apply(&init/11, init_args ++ [defn_options]) end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.client_preprocessing(&client_preprocessing(&1, tokenizer))
    |> Nx.Serving.client_postprocessing(&client_postprocessing(&1, &2, safety_checker))
  end

  defp init(
         {image_fun, safety_checker_fun},
         encoder_params,
         unet_params,
         vae_params,
         controlnet_params,
         {safety_checker?, safety_checker_spec, safety_checker_params},
         safety_checker_featurizer,
         {compile?, batch_size, sequence_length, conditioning_size},
         num_images_per_prompt,
         preallocate_params,
         defn_options
       ) do
    encoder_params = Shared.maybe_preallocate(encoder_params, preallocate_params, defn_options)
    unet_params = Shared.maybe_preallocate(unet_params, preallocate_params, defn_options)
    vae_params = Shared.maybe_preallocate(vae_params, preallocate_params, defn_options)

    controlnet_params =
      Shared.maybe_preallocate(controlnet_params, preallocate_params, defn_options)

    image_fun =
      Shared.compile_or_jit(image_fun, defn_options, compile?, fn ->
        inputs = %{
          "conditional_and_unconditional" => %{
            "input_ids" => Nx.template({batch_size, 2, sequence_length}, :u32)
          },
          "seed" => Nx.template({batch_size}, :s64),
          "conditioning" =>
            Nx.template(
              {batch_size, conditioning_size, conditioning_size, 3},
              :f32
            ),
          "conditioning_scale" => Nx.template({batch_size}, :f32)
        }

        [encoder_params, unet_params, vae_params, controlnet_params, inputs]
      end)

    safety_checker_fun =
      safety_checker_fun &&
        Shared.compile_or_jit(safety_checker_fun, defn_options, compile?, fn ->
          inputs = %{
            "pixel_values" =>
              Shared.input_template(safety_checker_spec, "pixel_values", [
                batch_size * num_images_per_prompt
              ])
          }

          [safety_checker_params, inputs]
        end)

    safety_checker_params =
      safety_checker_params &&
        Shared.maybe_preallocate(safety_checker_params, preallocate_params, defn_options)

    fn inputs ->
      inputs = Shared.maybe_pad(inputs, batch_size)

      image = image_fun.(encoder_params, unet_params, vae_params, controlnet_params, inputs)

      output =
        if safety_checker? do
          inputs = Bumblebee.apply_featurizer(safety_checker_featurizer, image)
          outputs = safety_checker_fun.(safety_checker_params, inputs)
          %{image: image, is_unsafe: outputs.is_unsafe}
        else
          %{image: image}
        end

      output
      |> Utils.Nx.composite_unflatten_batch(Utils.Nx.batch_size(inputs))
      |> Shared.serving_post_computation()
    end
  end

  defp preprocess_image(image) do
    NxImage.to_continuous(image, 0, 1)
  end

  defp client_preprocessing(input, tokenizer) do
    {inputs, multi?} = Shared.validate_serving_input!(input, &validate_input/1)

    seed = Enum.map(inputs, & &1.seed) |> Nx.tensor(backend: Nx.BinaryBackend)

    # Note: we need to tokenize all sequences together, so that
    # they are padded to the same length (if not specified)
    prompts = Enum.flat_map(inputs, &[&1.prompt, &1.negative_prompt])

    prompt_pairs =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        inputs = Bumblebee.apply_tokenizer(tokenizer, prompts)
        Utils.Nx.composite_unflatten_batch(inputs, Nx.axis_size(seed, 0))
      end)

    conditioning =
      Enum.map(inputs, & &1.conditioning)
      |> Nx.stack()
      |> preprocess_image()

    conditioning_scale =
      Enum.map(inputs, & &1.conditioning_scale)
      |> Nx.tensor(type: :f32, backend: Nx.BinaryBackend)

    inputs = %{
      "conditional_and_unconditional" => prompt_pairs,
      "seed" => seed,
      "conditioning" => conditioning,
      "conditioning_scale" => conditioning_scale
    }

    {Nx.Batch.concatenate([inputs]), multi?}
  end

  defp client_postprocessing({outputs, _metadata}, multi?, safety_checker?) do
    for outputs <- Utils.Nx.batch_to_list(outputs) do
      results =
        for outputs = %{image: image} <- Utils.Nx.batch_to_list(outputs) do
          if safety_checker? do
            if Nx.to_number(outputs.is_unsafe) == 1 do
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
    |> Nx.tensor(type: Nx.type(tensor), backend: Nx.BinaryBackend)
    |> Nx.broadcast(Nx.shape(tensor))
  end

  defnp text_to_image_impl(
          encoder_predict,
          encoder_params,
          unet_predict,
          unet_params,
          vae_predict,
          vae_params,
          controlnet_predict,
          controlnet_params,
          scheduler_init,
          scheduler_step,
          inputs,
          opts \\ []
        ) do
    num_images_per_prompt = opts[:num_images_per_prompt]
    latents_sample_size = opts[:latents_sample_size]
    latents_in_channels = opts[:latents_channels]
    guidance_scale = opts[:guidance_scale]

    seed = inputs["seed"]
    conditioning = inputs["conditioning"]
    conditioning_scale = inputs["conditioning_scale"]

    inputs =
      inputs["conditional_and_unconditional"]
      # Transpose conditional and unconditional to separate blocks
      |> composite_transpose_leading()
      |> Utils.Nx.composite_flatten_batch()

    %{hidden_state: text_embeddings} = encoder_predict.(encoder_params, inputs)

    {_twice_batch_size, sequence_length, hidden_size} = Nx.shape(text_embeddings)

    text_embeddings =
      text_embeddings
      |> Nx.new_axis(1)
      |> Nx.tile([1, num_images_per_prompt, 1, 1])
      |> Nx.reshape({:auto, sequence_length, hidden_size})

    prng_key =
      seed
      |> Nx.vectorize(:batch)
      |> Nx.Random.key()
      |> Nx.Random.split(parts: num_images_per_prompt)
      |> Nx.devectorize()
      |> Nx.flatten(axes: [0, 1])
      |> Nx.vectorize(:batch)

    {latents, prng_key} =
      Nx.Random.normal(prng_key,
        shape: {latents_sample_size, latents_sample_size, latents_in_channels}
      )

    {scheduler_state, timesteps} = scheduler_init.(Nx.to_template(latents), prng_key)

    latents = Nx.devectorize(latents)

    {latents, _} =
      while {latents,
             {scheduler_state, text_embeddings, unet_params, conditioning, conditioning_scale,
              controlnet_params}},
            timestep <- timesteps do
        controlnet_inputs = %{
          "conditioning" => conditioning,
          "conditioning_scale" => conditioning_scale,
          "sample" => Nx.concatenate([latents, latents]),
          "timestep" => timestep,
          "encoder_hidden_state" => text_embeddings
        }

        %{down_block_states: down_block_states, mid_block_state: mid_block_state} =
          controlnet_predict.(controlnet_params, controlnet_inputs)

        unet_inputs =
          %{
            "sample" => Nx.concatenate([latents, latents]),
            "timestep" => timestep,
            "encoder_hidden_state" => text_embeddings,
            "additional_down_block_states" => down_block_states,
            "additional_mid_block_state" => mid_block_state
          }

        %{sample: noise_pred} = unet_predict.(unet_params, unet_inputs)

        {noise_pred_conditional, noise_pred_unconditional} =
          split_conditional_and_unconditional(noise_pred)

        noise_pred =
          noise_pred_unconditional +
            guidance_scale * (noise_pred_conditional - noise_pred_unconditional)

        {scheduler_state, latents} =
          scheduler_step.(
            scheduler_state,
            Nx.vectorize(latents, :batch),
            Nx.vectorize(noise_pred, :batch)
          )

        latents = Nx.devectorize(latents)

        {latents,
         {scheduler_state, text_embeddings, unet_params, conditioning, conditioning_scale,
          controlnet_params}}
      end

    latents = latents * (1 / 0.18215)

    %{sample: image} = vae_predict.(vae_params, latents)

    NxImage.from_continuous(image, -1, 1)
  end

  deftransformp composite_transpose_leading(container) do
    Utils.Nx.map(container, fn tensor ->
      [first, second | rest] = Nx.axes(tensor)
      Nx.transpose(tensor, axes: [second, first | rest])
    end)
  end

  defnp split_conditional_and_unconditional(tensor) do
    batch_size = Nx.axis_size(tensor, 0)
    half_size = div(batch_size, 2)
    {tensor[0..(half_size - 1)//1], tensor[half_size..-1//1]}
  end

  defp validate_input(prompt) when is_binary(prompt), do: validate_input(%{prompt: prompt})

  defp validate_input(%{prompt: prompt, conditioning: conditioning} = input) do
    {:ok,
     %{
       prompt: prompt,
       conditioning: conditioning,
       conditioning_scale: input[:conditioning_scale] || 1.0,
       negative_prompt: input[:negative_prompt] || "",
       seed: input[:seed] || :erlang.system_time()
     }}
  end

  defp validate_input(%{} = input) do
    {:error,
     "expected the input map to have :prompt and :conditioning key, got: #{inspect(input)}"}
  end

  defp validate_input(input) do
    {:error, "expected either a string or a map, got: #{inspect(input)}"}
  end
end
