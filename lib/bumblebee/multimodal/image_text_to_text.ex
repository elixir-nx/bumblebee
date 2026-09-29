defmodule Bumblebee.Multimodal.ImageTextToText do
  @moduledoc """
  Generation helpers for vision-language models like Qwen3-VL.

  Two entry points:

    * `generate/6` — one-shot call. Featurizes, expands the prompt
      placeholder, and runs generation. Each call recompiles the graph
      when the image or sequence length changes, so it suits
      interactive use.

    * `compile/5` + `run/3` — compile the generation graph **once** for
      upper-bound shapes, then run repeatedly with images of varying
      sizes. The featurizer pads `pixel_values` and `image_grid_thw` to
      the configured maxima, and the vision encoder excludes padded
      patches from attention via `patch_valid`.
  """

  alias Bumblebee.Text

  @placeholder "<|image_pad|>"

  @doc """
  Generates text from a prompt that includes a `<|image_pad|>` marker
  and an image.

  ## Required arguments

    * `model_info` - a loaded `Bumblebee.Multimodal.Qwen3VL` (or compatible)
      model
    * `featurizer` - a configured `Bumblebee.Vision.Qwen3VLFeaturizer`
    * `tokenizer` - a loaded tokenizer for the same model
    * `generation_config` - a `Bumblebee.Text.GenerationConfig`
    * `text` - the user prompt containing exactly one `<|image_pad|>` marker
    * `image` - an image tensor or `t:StbImage.t/0`

  ## Returns

      %{text: "<generated text>", token_ids: [...]}

  ## Example

      {:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-VL-2B-Instruct"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-VL-2B-Instruct"})

      {:ok, featurizer} =
        Bumblebee.load_featurizer({:hf, "Qwen/Qwen3-VL-2B-Instruct"},
          module: Bumblebee.Vision.Qwen3VLFeaturizer
        )

      featurizer = Bumblebee.configure(featurizer, quality: :low)
      {:ok, gen_config} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-VL-2B-Instruct"})
      gen_config = Bumblebee.configure(gen_config, max_new_tokens: 64)

      Bumblebee.Multimodal.ImageTextToText.generate(
        model_info, featurizer, tokenizer, gen_config,
        "<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>What is in this image?<|im_end|>\\n<|im_start|>assistant\\n",
        image
      )
  """
  def generate(
        model_info,
        featurizer,
        tokenizer,
        %Text.GenerationConfig{} = generation_config,
        text,
        image
      ) do
    %{model: model, params: params, spec: spec} = model_info

    unless Map.has_key?(spec, :image_token_id) do
      raise ArgumentError,
            "expected a multimodal model with :image_token_id, got #{inspect(spec.__struct__)}"
    end

    merge_size =
      case spec do
        %{vision_spec: %{spatial_merge_size: ms}} -> ms
        _ -> 1
      end

    image_inputs = Bumblebee.apply_featurizer(featurizer, image)
    visual_tokens = visual_tokens_for(image_inputs["image_grid_thw"], merge_size)
    expanded_text = expand_marker(text, visual_tokens)

    tokenizer = Bumblebee.configure(tokenizer, return_token_type_ids: false)
    text_inputs = Bumblebee.apply_tokenizer(tokenizer, expanded_text)

    inputs =
      text_inputs
      |> Map.merge(image_inputs)
      |> Map.put("seed", Nx.tensor([:erlang.system_time()], type: :s64))

    generate_fun = Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate_fun.(params, inputs)

    decoded =
      token_ids
      |> Nx.to_batched(1)
      |> Enum.map(&Bumblebee.Tokenizer.decode(tokenizer, Nx.to_flat_list(&1)))
      |> hd()

    %{text: decoded, token_ids: token_ids}
  end

  @doc """
  Compiles the generation graph once for the given upper-bound shapes.

  The returned struct can be passed to `run/3` repeatedly. Calls with
  images that produce fewer than `:max_patches` real patches or
  shorter than `:sequence_length` prompts are padded; the vision
  encoder masks the padded positions out of attention.

  ## Options

    * `:max_patches` (required) — upper bound on total patches across
      all images in one call. Must be a multiple of `merge_size ** 2`.
    * `:max_num_images` (required) — upper bound on number of images
      per call.
    * `:sequence_length` (required) — upper bound on token count
      (prompt + generated).
  """
  def compile(
        model_info,
        featurizer,
        tokenizer,
        %Text.GenerationConfig{} = generation_config,
        opts
      ) do
    opts = Keyword.validate!(opts, [:max_patches, :max_num_images, :sequence_length])
    max_patches = Keyword.fetch!(opts, :max_patches)
    max_num_images = Keyword.fetch!(opts, :max_num_images)
    sequence_length = Keyword.fetch!(opts, :sequence_length)

    %{model: model, params: params, spec: spec} = model_info

    unless Map.has_key?(spec, :image_token_id) do
      raise ArgumentError,
            "expected a multimodal model with :image_token_id, got #{inspect(spec.__struct__)}"
    end

    merge_size = spec.vision_spec.spatial_merge_size

    featurizer =
      Bumblebee.configure(featurizer,
        max_patches: max_patches,
        max_num_images: max_num_images
      )

    tokenizer =
      Bumblebee.configure(tokenizer,
        length: sequence_length,
        pad_direction: :left,
        return_token_type_ids: false
      )

    generate_fun = Text.Generation.build_generate(model, spec, generation_config)

    %{
      generate_fun: generate_fun,
      params: params,
      spec: spec,
      featurizer: featurizer,
      tokenizer: tokenizer,
      merge_size: merge_size,
      max_patches: max_patches,
      max_num_images: max_num_images,
      sequence_length: sequence_length
    }
  end

  @doc """
  Runs a prompt + image through a pre-compiled generator from `compile/5`.

  EXLA caches the compiled graph by input shape; since the featurizer
  pads to the upper bounds configured in `compile/5`, every call hits
  the same cached graph.
  """
  def run(compiled, text, image) do
    %{
      generate_fun: generate_fun,
      params: params,
      featurizer: featurizer,
      tokenizer: tokenizer,
      merge_size: merge_size
    } = compiled

    image_inputs = Bumblebee.apply_featurizer(featurizer, image)
    grid_thw_real = unpad_grid_thw(image_inputs["image_grid_thw"])
    visual_tokens = visual_tokens_for(grid_thw_real, merge_size)
    expanded_text = expand_marker(text, visual_tokens)

    text_inputs = Bumblebee.apply_tokenizer(tokenizer, expanded_text)

    inputs =
      text_inputs
      |> Map.merge(image_inputs)
      |> Map.put("seed", Nx.tensor([:erlang.system_time()], type: :s64))

    %{token_ids: token_ids} = generate_fun.(params, inputs)

    decoded =
      token_ids
      |> Nx.to_batched(1)
      |> Enum.map(&Bumblebee.Tokenizer.decode(tokenizer, Nx.to_flat_list(&1)))
      |> hd()

    %{text: decoded, token_ids: token_ids}
  end

  # Drops padding rows ([0, 0, 0]) so visual_tokens_for matches the
  # actual prompt expansion length.
  defp unpad_grid_thw(grid_thw) do
    grid_thw
    |> Nx.to_list()
    |> Enum.reject(fn [t, h, w] -> t == 0 and h == 0 and w == 0 end)
    |> case do
      [] -> Nx.tensor([[0, 0, 0]], type: :s64)
      rows -> Nx.tensor(rows, type: :s64)
    end
  end

  defp expand_marker(text, visual_tokens) do
    case String.split(text, @placeholder) do
      [_only] ->
        raise ArgumentError,
              "the prompt must contain a #{@placeholder} marker where the image " <>
                "embedding should be spliced in, got: #{inspect(text)}"

      [prefix, suffix] ->
        prefix <> String.duplicate(@placeholder, visual_tokens) <> suffix

      _multiple ->
        raise ArgumentError,
              "expected exactly one #{@placeholder} marker in the prompt"
    end
  end

  defp visual_tokens_for(grid_thw, merge_size) do
    grid_thw
    |> Nx.to_list()
    |> Enum.map(fn [t, h, w] ->
      t * div(h, merge_size) * div(w, merge_size)
    end)
    |> Enum.sum()
  end
end
