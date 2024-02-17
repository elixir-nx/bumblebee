defmodule Bumblebee.Diffusion.UNet2DConditionalTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-stable-diffusion-torch", subdir: "unet"}
             )

    assert %Bumblebee.Diffusion.UNet2DConditional{architecture: :base} = spec

    inputs = %{
      "sample" => Nx.broadcast(0.5, {1, 32, 32, 4}),
      "timestep" => Nx.tensor(1),
      "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 32})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.sample) == {1, 32, 32, 4}

    assert_all_close(
      to_channels_first(outputs.sample)[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [
            [-1.0813, -0.5109, -0.1545],
            [-0.8094, -1.2588, -0.8355],
            [-0.9218, -1.2142, -0.6982]
          ],
          [
            [-0.2179, -0.2799, -1.0922],
            [-0.9485, -0.8376, 0.0843],
            [-0.9650, -0.7105, -0.3920]
          ],
          [[1.3359, 0.8373, -0.2392], [0.9448, -0.0478, 0.6881], [-0.0154, -0.5304, 0.2081]]
        ]
      ]),
      atol: 1.0e-4
    )
  end

  test ":with_controlnet" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-stable-diffusion-torch", subdir: "unet"},
               architecture: :with_controlnet
             )

    assert %Bumblebee.Diffusion.UNet2DConditional{architecture: :with_controlnet} = spec

    num_down_residuals = (length(spec.hidden_sizes) * (spec.depth + 1)) |> dbg()

    out_channels = [32, 32, 32, 32, 64, 64]
    out_spatials = [32, 32, 32, 16, 16, 16]

    down_zip = Enum.zip(out_channels, out_spatials)

    down_residuals =
      for {{out_channel, out_spatial}, i} <- Enum.with_index(down_zip), into: %{} do
        shape = {1, out_spatial, out_spatial, out_channel}
        {"controlnet_down_residual_#{i}", Nx.broadcast(0.5, shape)}
      end

    inputs =
      %{
        "sample" => Nx.broadcast(0.5, {1, 32, 32, 4}),
        "timestep" => Nx.tensor(1),
        "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 32}),
        "controlnet_mid_residual" => Nx.broadcast(0.5, {1, 16, 16, 64})
      }
      |> Map.merge(down_residuals)

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.sample) == {1, 32, 32, 4}

    assert_all_close(
      to_channels_first(outputs.sample)[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [
            [-1.0813, -0.5109, -0.1545],
            [-0.8094, -1.2588, -0.8355],
            [-0.9218, -1.2142, -0.6982]
          ],
          [
            [-0.2179, -0.2799, -1.0922],
            [-0.9485, -0.8376, 0.0843],
            [-0.9650, -0.7105, -0.3920]
          ],
          [[1.3359, 0.8373, -0.2392], [0.9448, -0.0478, 0.6881], [-0.0154, -0.5304, 0.2081]]
        ]
      ]),
      atol: 1.0e-4
    )
  end
end
