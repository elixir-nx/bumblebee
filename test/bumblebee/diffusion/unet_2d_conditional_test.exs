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

    first = {1, spec.sample_size, spec.sample_size, hd(spec.hidden_sizes)}

    state = {spec.sample_size, [first]}

    {_, out_shapes} =
      for block_out_channel <- spec.hidden_sizes, reduce: state do
        {spatial_size, acc} ->
          states =
            for _ <- 1..spec.depth, do: {1, spatial_size, spatial_size, block_out_channel}

          downsampled_spatial = div(spatial_size, 2)
          downsample = {1, downsampled_spatial, downsampled_spatial, block_out_channel}

          {div(spatial_size, 2), acc ++ states ++ [downsample]}
      end

    out_shapes = Enum.drop(out_shapes, -1)

    down_block_states =
      for shape <- out_shapes do
        Nx.broadcast(0.5, shape)
      end
      |> List.to_tuple()

    mid_spatial = div(spec.sample_size, 2 ** (length(spec.hidden_sizes) - 1))
    mid_dim = List.last(spec.hidden_sizes)
    mid_block_shape = {1, mid_spatial, mid_spatial, mid_dim}

    inputs =
      %{
        "sample" => Nx.broadcast(0.5, {1, spec.sample_size, spec.sample_size, 4}),
        "timestep" => Nx.tensor(1),
        "encoder_hidden_state" => Nx.broadcast(0.5, {1, 1, spec.cross_attention_size}),
        "additional_mid_block_state" => Nx.broadcast(0.5, mid_block_shape),
        "additional_down_block_states" => down_block_states
      }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.sample) == {1, spec.sample_size, spec.sample_size, spec.in_channels}

    assert_all_close(
      outputs.sample[[.., 1..3, 1..3, 1..3]],
      Nx.tensor([
        [
          [-2.1599538326263428, -0.06256292015314102, 1.7675844430923462],
          [-0.6707635521888733, -0.6823181509971619, 1.0919926166534424],
          [0.16482116281986237, -1.2743796110153198, -0.03096655011177063]
        ],
        [
          [-1.13632333278656, -1.3499518632888794, 0.597271203994751],
          [-1.7593439817428589, -1.599103569984436, -0.1870473176240921],
          [-0.9655789136886597, 0.8080697655677795, 1.1974149942398071]
        ],
        [
          [-1.3559075593948364, -1.177065134048462, -0.5016229152679443],
          [-2.5425026416778564, -1.2682275772094727, -0.6805112957954407],
          [-1.8208105564117432, 0.9214832186698914, 0.5924324989318848]
        ]
      ]),
      atol: 1.0e-4
    )
  end
end
