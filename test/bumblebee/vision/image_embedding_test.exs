defmodule Bumblebee.Vision.ImageEmbeddingTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()
  @images_dir Path.expand("../../fixtures/images", __DIR__)

  test "returns embedding for an image" do
    {:ok, model_info} =
      Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"},
        module: Bumblebee.Vision.ClipVision
      )

    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/clip-vit-base-patch32"})

    serving = Bumblebee.Vision.ImageEmbedding.image_embedding(model_info, featurizer)
    image = StbImage.read_file!(Path.join(@images_dir, "coco/39769.jpeg"))

    assert %{embedding: %Nx.Tensor{} = embedding} = Nx.Serving.run(serving, image)
    assert Nx.shape(embedding) == {768}

    assert_all_close(
      embedding[1..3],
      Nx.tensor([0.0978, -0.7233, -0.7707])
    )
  end

  test "returns normalized embedding for an image" do
    {:ok, model_info} =
      Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"},
        module: Bumblebee.Vision.ClipVision
      )

    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/clip-vit-base-patch32"})

    options = [
      embedding_processor: :l2_norm
    ]

    serving = Bumblebee.Vision.ImageEmbedding.image_embedding(model_info, featurizer, options)
    image = StbImage.read_file!(Path.join(@images_dir, "coco/39769.jpeg"))

    assert %{embedding: %Nx.Tensor{} = embedding} = Nx.Serving.run(serving, image)
    assert Nx.shape(embedding) == {768}

    assert_all_close(
      embedding[1..3],
      Nx.tensor([0.0036, -0.0269, -0.0286])
    )
  end
end
