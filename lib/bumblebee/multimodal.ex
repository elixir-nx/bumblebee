defmodule Bumblebee.Multimodal do
  @moduledoc """
  High-level tasks related to multimodal input.
  """

  @type image_to_text_input :: Bumblebee.Vision.image()
  @type image_to_text_output :: %{results: list(image_to_text_result())}
  @type image_to_text_result :: %{text: String.t()}

  @doc """
  Builds serving for image-to-text generation.

  The serving accepts `t:image_to_text_input/0` and returns
  `t:image_to_text_output/0`. A list of inputs is also supported.

  Note that either `:max_new_tokens` or `:max_length` must be specified.
  The generation should generally finish based on the audio input,
  however you still need to specify the upper limit.

  ## Options

    * `:max_new_tokens` - the maximum number of tokens to be generated,
      ignoring the number of tokens in the prompt

    * `:compile` - compiles all computations for predefined input shapes
      during serving initialization. Should be a keyword list with the
      following keys:

        * `:batch_size` - the maximum batch size of the input. Inputs
          are optionally padded to always match this batch size

      It is advised to set this option in production and also configure
      a defn compiler using `:defn_options` to maximally reduce inference
      time.

    * `:defn_options` - the options for JIT compilation. Defaults to `[]`

  Also accepts all the other options of `Bumblebee.Text.Generation.build_generate/3`.

  ## Examples

      {:ok, blip} = Bumblebee.load_model({:hf, "Salesforce/blip-image-captioning-base"})
      {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "Salesforce/blip-image-captioning-base"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Salesforce/blip-image-captioning-large"})

      serving =
        Bumblebee.Multimodal.ImageToText.image_to_text(blip, featurizer, tokenizer,
          max_new_tokens: 100,
          defn_options: [compiler: EXLA]
        )

      image = StbImage.read_file!(path)
      Nx.Serving.run(serving, image)
      #=> %{results: [%{text: "a cat sitting on a chair"}]}

  """
  @spec image_to_text(
          Bumblebee.model_info(),
          Bumblebee.Featurizer.t(),
          Bumblebee.Tokenizer.t(),
          keyword()
        ) :: Nx.Serving.t()
  defdelegate image_to_text(model_info, featurizer, tokenizer, opts \\ []),
    to: Bumblebee.Multimodal.ImageToText
end
