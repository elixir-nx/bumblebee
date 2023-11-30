defmodule Bumblebee.Vision.ImageClassificationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  @images_dir Path.expand("../../fixtures/images", __DIR__)

  test "returns top scored labels" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "microsoft/resnet-50"})

    serving = Bumblebee.Vision.ImageClassification.image_classification(model_info, featurizer)

    image = StbImage.read_file!(Path.join(@images_dir, "coco/39769.jpeg"))

    assert %{
             predictions: [
               %{label: "tiger cat", score: _},
               %{label: "tabby, tabby cat", score: _},
               %{label: "remote control, remote", score: _},
               %{label: "jinrikisha, ricksha, rickshaw", score: _},
               %{label: "Egyptian cat", score: _}
             ]
           } = Nx.Serving.run(serving, image)
  end

  test "supports compilation" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "microsoft/resnet-50"})

    serving =
      Bumblebee.Vision.ImageClassification.image_classification(model_info, featurizer,
        compile: [batch_size: 1],
        defn_options: [compiler: EXLA]
      )

    image = StbImage.read_file!(Path.join(@images_dir, "coco/39769.jpeg"))

    assert %{
             predictions: [
               %{label: "tiger cat", score: _},
               %{label: "tabby, tabby cat", score: _},
               %{label: "remote control, remote", score: _},
               %{label: "jinrikisha, ricksha, rickshaw", score: _},
               %{label: "Egyptian cat", score: _}
             ]
           } = Nx.Serving.run(serving, image)
  end
end
