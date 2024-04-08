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
      outputs.sample[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [[-1.0813, -0.2179, 1.3359], [-0.5109, -0.2799, 0.8373], [-0.1545, -1.0922, -0.2392]],
          [[-0.8094, -0.9485, 0.9448], [-1.2588, -0.8376, -0.0478], [-0.8355, 0.0843, 0.6881]],
          [[-0.9218, -0.9650, -0.0154], [-1.2142, -0.7105, -0.5304], [-0.6982, -0.3920, 0.2081]]
        ]
      ])
    )
  end

  test ":base with additional states for skip connection" do
    tiny = "bumblebee-testing/tiny-stable-diffusion"

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, tiny, subdir: "unet"})

    assert %Bumblebee.Diffusion.UNet2DConditional{architecture: :base} = spec

    down_block_states =
      [
        {1, 32, 32, 32},
        {1, 32, 32, 32},
        {1, 32, 32, 32},
        {1, 16, 16, 32},
        {1, 16, 16, 64},
        {1, 16, 16, 64}
      ]
      |> Enum.map(&Nx.broadcast(0.5, &1))
      |> List.to_tuple()

    mid_block_state = Nx.broadcast(0.5, {1, 16, 16, 64})

    inputs =
      %{
        "sample" => Nx.broadcast(0.5, {1, 32, 32, 4}),
        "timestep" => Nx.tensor(1),
        "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, 32}),
        "additional_down_block_states" => down_block_states,
        "additional_mid_block_state" => mid_block_state
      }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.sample) == {1, 32, 32, 4}

    assert_all_close(
      outputs.sample[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [[-0.9457, -0.2378, 1.4223], [-0.5736, -0.2456, 0.7603], [-0.4346, -1.1370, -0.1988]],
          [[-0.5274, -1.0902, 0.5937], [-1.2290, -0.7996, 0.0264], [-0.3006, -0.1181, 0.7059]],
          [[-0.8336, -1.1615, -0.1906], [-1.0489, -0.3815, -0.5497], [-0.6255, 0.0863, 0.3285]]
        ]
      ])
    )
  end
end
