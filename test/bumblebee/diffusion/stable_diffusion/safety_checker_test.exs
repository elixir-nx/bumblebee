defmodule Bumblebee.Diffusion.StableDiffusion.SafetyCheckerTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  # Note that here we use a full-sized model because the output is a
  # binary answer and we want to validate that it actually works
  @moduletag slow: true, timeout: 600_000

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "CompVis/stable-diffusion-v1-4", subdir: "safety_checker"}
             )

    assert {:ok, featurizer} =
             Bumblebee.load_featurizer(
               {:hf, "CompVis/stable-diffusion-v1-4", subdir: "feature_extractor"}
             )

    assert %Bumblebee.Diffusion.StableDiffusion.SafetyChecker{architecture: :base} = spec

    safe1 = Nx.broadcast(100, {1, 224, 224, 3})

    safe2 =
      Nx.tensor([
        [
          [204, 187, 172],
          [165, 142, 130],
          [216, 183, 169],
          [226, 202, 183],
          [220, 190, 178],
          [180, 153, 148]
        ],
        [
          [213, 195, 179],
          [113, 96, 86],
          [102, 91, 85],
          [124, 111, 105],
          [117, 97, 93],
          [78, 72, 71]
        ],
        [
          [203, 187, 172],
          [90, 81, 75],
          [83, 53, 50],
          [83, 54, 51],
          [70, 56, 53],
          [127, 109, 107]
        ],
        [
          [221, 210, 192],
          [95, 83, 79],
          [127, 65, 66],
          [128, 63, 63],
          [90, 70, 69],
          [186, 163, 161]
        ],
        [
          [227, 221, 198],
          [131, 122, 116],
          [192, 152, 147],
          [180, 137, 126],
          [143, 121, 114],
          [204, 175, 172]
        ],
        [
          [178, 172, 156],
          [89, 83, 78],
          [156, 145, 136],
          [116, 104, 95],
          [96, 77, 73],
          [176, 144, 142]
        ]
      ])

    unsafe1 =
      Nx.tensor([
        [
          [158, 106, 56],
          [222, 147, 81],
          [233, 180, 125],
          [226, 164, 105],
          [244, 205, 162],
          [189, 170, 148]
        ],
        [
          [180, 124, 68],
          [233, 163, 88],
          [232, 183, 130],
          [213, 145, 88],
          [244, 196, 145],
          [224, 203, 178]
        ],
        [
          [202, 158, 108],
          [228, 168, 103],
          [220, 162, 108],
          [190, 133, 88],
          [223, 168, 118],
          [229, 206, 182]
        ],
        [
          [172, 135, 106],
          [237, 197, 160],
          [241, 203, 171],
          [236, 192, 153],
          [224, 178, 138],
          [141, 109, 89]
        ],
        [
          [156, 112, 88],
          [228, 179, 139],
          [234, 188, 153],
          [225, 171, 132],
          [217, 163, 124],
          [123, 86, 67]
        ],
        [
          [133, 88, 66],
          [181, 113, 75],
          [151, 93, 63],
          [150, 89, 61],
          [162, 94, 63],
          [100, 63, 47]
        ]
      ])

    unsafe2 =
      Nx.tensor([
        [
          [148, 120, 67],
          [152, 123, 70],
          [136, 109, 59],
          [114, 88, 44],
          [171, 114, 69],
          [233, 135, 85]
        ],
        [
          [158, 126, 72],
          [209, 139, 87],
          [211, 139, 87],
          [211, 144, 91],
          [247, 161, 107],
          [243, 148, 96]
        ],
        [
          [187, 136, 82],
          [223, 138, 88],
          [249, 119, 82],
          [251, 99, 73],
          [249, 138, 97],
          [249, 162, 111]
        ],
        [
          [245, 160, 106],
          [244, 167, 112],
          [241, 128, 90],
          [233, 91, 67],
          [226, 103, 73],
          [215, 113, 72]
        ],
        [
          [168, 79, 45],
          [173, 76, 43],
          [171, 56, 33],
          [162, 41, 25],
          [150, 35, 20],
          [142, 44, 24]
        ],
        [[86, 52, 24], [111, 66, 33], [109, 61, 30], [104, 48, 21], [103, 46, 20], [72, 32, 17]]
      ])

    # Note: the example images are downscaled to 6x6 as it appears
    # to be enough for this test case and this way we don't need to
    # keep unsafe images around
    inputs = Bumblebee.apply_featurizer(featurizer, [safe1, safe2, unsafe1, unsafe2])

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.is_unsafe) == {4}

    assert_all_close(
      outputs.is_unsafe,
      Nx.tensor([0, 0, 1, 1]),
      atol: 1.0e-4
    )
  end
end
