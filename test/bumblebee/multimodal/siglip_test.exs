defmodule Bumblebee.Multimodal.SigLipTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "katuni4ka/tiny-random-SiglipModel"})

    assert %Bumblebee.Multimodal.SigLip{architecture: :base} = spec

    # Image size is 30x30 for this tiny model
    inputs = %{
      "input_ids" =>
        Nx.tensor([
          [10, 20, 30, 40, 50, 60, 70, 80, 1, 1],
          [15, 25, 35, 45, 55, 65, 75, 85, 1, 1]
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
      Nx.tensor([[-0.0626, -0.0771], [-0.0961, -0.1548]]),
      atol: 1.0e-3
    )

    assert_all_close(
      outputs.logits_per_image,
      Nx.tensor([[-0.0626, -0.0961], [-0.0771, -0.1548]]),
      atol: 1.0e-3
    )
  end
end
