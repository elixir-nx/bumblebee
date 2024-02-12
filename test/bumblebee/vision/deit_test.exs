defmodule Bumblebee.Vision.DeitTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DeiTModel"})

    assert %Bumblebee.Vision.Deit{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 227, 32}
    assert Nx.shape(outputs.pooled_state) == {1, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-3.0866, 0.2350, 0.2003], [-1.2774, -0.1192, -1.0468], [-1.2774, -0.1192, -1.0468]]
      ])
    )

    assert_all_close(
      outputs.pooled_state[[.., 1..3]],
      Nx.tensor([[0.1526, -0.1437, -0.0646]])
    )
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DeiTForImageClassification"}
             )

    assert %Bumblebee.Vision.Deit{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[0.0481, 0.1008]])
    )
  end

  test ":for_image_classification_with_teacher" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DeiTForImageClassificationWithTeacher"}
             )

    assert %Bumblebee.Vision.Deit{architecture: :for_image_classification_with_teacher} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.0108, -0.0048]])
    )
  end

  test ":for_masked_image_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DeiTForMaskedImageModeling"}
             )

    assert %Bumblebee.Vision.Deit{architecture: :for_masked_image_modeling} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.pixel_values) == {1, 30, 30, 3}

    assert_all_close(
      to_channels_first(outputs.pixel_values)[[.., 1..2, 1..2, 1..2]],
      Nx.tensor([[[[0.1455, 0.0229], [-0.0097, 0.0525]], [[0.1889, 0.0910], [-0.1083, -0.0244]]]])
    )
  end
end
