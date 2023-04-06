defmodule Bumblebee.Vision.ImageToTextTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @images_dir Path.expand("../../fixtures/images", __DIR__)

  describe "integration" do
    test "returns top scored labels" do
      {:ok, blip} = Bumblebee.load_model({:hf, "Salesforce/blip-image-captioning-base"})

      {:ok, featurizer} =
        Bumblebee.load_featurizer({:hf, "Salesforce/blip-image-captioning-base"})

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Salesforce/blip-image-captioning-base"})

      {:ok, generation_config} =
        Bumblebee.load_generation_config({:hf, "Salesforce/blip-image-captioning-base"})

      serving =
        Bumblebee.Vision.ImageToText.image_to_text(blip, featurizer, tokenizer, generation_config,
          defn_options: [compiler: EXLA]
        )

      image = StbImage.read_file!(Path.join(@images_dir, "coco/39769.jpeg"))

      assert %{
               results: [%{text: "two cats are laying on a pink couch"}]
             } = Nx.Serving.run(serving, image)
    end
  end
end
