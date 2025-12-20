defmodule Bumblebee.Vision.PixtralTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:local, "test/fixtures/models/tiny-random-PixtralVisionModel"})

    assert %Bumblebee.Vision.Pixtral{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    # With default patch_size of 14 and image_size of 224:
    # num_patches = (224 / 14)^2 = 16^2 = 256
    assert Nx.shape(outputs.hidden_state) == {1, 256, 32}

    # Expected values from Bumblebee (see test/fixtures/scripts/bumblebee_expected_values.txt)
    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [
          [-0.3103204071521759, 0.19613781571388245, -0.6223983764648438],
          [-0.3103204071521759, 0.19613781571388245, -0.6223983764648438],
          [-0.3103204071521759, 0.19613780081272125, -0.6223983764648438]
        ]
      ]),
      atol: 1.0e-4
    )
  end

  # Test that module structure and options are correct
  test "module structure" do
    assert Bumblebee.Vision.Pixtral.architectures() == [:base]

    # Test default configuration
    spec = %Bumblebee.Vision.Pixtral{}
    assert spec.architecture == :base
    assert spec.image_size == 1540
    assert spec.num_channels == 3
    assert spec.patch_size == 14
    assert spec.hidden_size == 1024
    assert spec.num_blocks == 24
    assert spec.num_attention_heads == 16
    assert spec.head_dim == 64
    assert spec.intermediate_size == 4096
    assert spec.activation == :silu
    assert spec.rotary_embedding_base == 10_000.0
  end

  test "configuration" do
    spec = %Bumblebee.Vision.Pixtral{}

    configured =
      Bumblebee.Vision.Pixtral.config(spec,
        image_size: 512,
        patch_size: 16
      )

    assert configured.image_size == 512
    assert configured.patch_size == 16
  end
end
