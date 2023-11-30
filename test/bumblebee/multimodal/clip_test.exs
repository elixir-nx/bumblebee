defmodule Bumblebee.Multimodal.ClipTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-CLIPModel"})

    assert %Bumblebee.Multimodal.Clip{architecture: :base} = spec

    inputs = %{
      "input_ids" =>
        Nx.tensor([
          [10, 20, 30, 40, 50, 60, 70, 80, 0, 0],
          [15, 25, 35, 45, 55, 65, 75, 85, 0, 0]
        ]),
      "attention_mask" =>
        Nx.tensor([
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        ]),
      "pixel_values" =>
        Nx.concatenate([
          Nx.broadcast(0.25, {1, 30, 30, 3}),
          Nx.broadcast(0.75, {1, 30, 30, 3})
        ])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits_per_text) == {2, 2}
    assert Nx.shape(outputs.logits_per_image) == {2, 2}

    assert_all_close(
      outputs.logits_per_text,
      Nx.tensor([[0.5381, 0.1981], [0.5212, 0.3291]]),
      atol: 1.0e-4
    )

    assert_all_close(
      outputs.logits_per_image,
      Nx.tensor([[0.5381, 0.5212], [0.1981, 0.3291]]),
      atol: 1.0e-4
    )
  end
end
