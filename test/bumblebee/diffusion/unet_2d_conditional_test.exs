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

  @tag timeout: :infinity
  test ":with_controlnet" do
    compvis = "CompVis/stable-diffusion-v1-4"
    tiny = "bumblebee-testing/tiny-stable-diffusion"

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, tiny, subdir: "unet"},
               architecture: :with_controlnet
             )

    assert %Bumblebee.Diffusion.UNet2DConditional{architecture: :with_controlnet} = spec

    first = {1, spec.sample_size, spec.sample_size, hd(spec.hidden_sizes)}

    state = {spec.sample_size, [first]}

    {mid_spatial, out_shapes} =
      for block_out_channel <- spec.hidden_sizes, reduce: state do
        {spatial_size, acc} ->
          residuals =
            for _ <- 1..spec.depth, do: {1, spatial_size, spatial_size, block_out_channel}

          downsampled_spatial = div(spatial_size, 2)
          downsample = {1, downsampled_spatial, downsampled_spatial, block_out_channel}

          {div(spatial_size, 2), acc ++ residuals ++ [downsample]}
      end

    mid_spatial = 2 * mid_spatial
    out_shapes = Enum.drop(out_shapes, -1) |> dbg()

    down_residuals =
      for {shape, i} <- Enum.with_index(out_shapes), into: %{} do
        {"controlnet_down_residual_#{i}", Nx.broadcast(0.5, shape)}
      end

    mid_dim = List.last(spec.hidden_sizes)

    mid_residual_shape = {1, mid_spatial, mid_spatial, mid_dim}

    inputs =
      %{
        "sample" => Nx.broadcast(0.5, {1, spec.sample_size, spec.sample_size, 4}),
        "timestep" => Nx.tensor(1),
        "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, spec.cross_attention_size}),
        "controlnet_mid_residual" => Nx.broadcast(0.5, mid_residual_shape)
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
