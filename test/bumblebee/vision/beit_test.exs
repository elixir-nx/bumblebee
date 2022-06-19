defmodule Bumblebee.Vision.BeitTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers
  require Axon

  describe "integration" do
    @tag :slow
    @tag :capture_log
    test "base model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "microsoft/beit-base-patch16-224"}, architecture: :base)

      assert %Bumblebee.Vision.Beit{architecture: :base} = config

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.pooler_output) == {1, 768}

      assert_all_close(
        Nx.sum(output.last_hidden_state),
        Nx.tensor(105_030.4297),
        atol: 1.0e-4
      )

      assert_all_close(
        Nx.sum(output.pooler_output),
        Nx.tensor(24.0872),
        atol: 1.0e-4
      )
    end

    @tag :slow
    test "image classification model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "microsoft/beit-base-patch16-224"})

      assert %Bumblebee.Vision.Beit{architecture: :for_image_classification} = config

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.logits) == {1, 1000}

      assert_all_close(
        output.logits[[0, 0..2]],
        Nx.tensor([-0.7950, -0.3834, -0.1417]),
        atol: 1.0e-4
      )
    end

    @tag :slow
    test "masked image modeling model" do
      assert {:ok, model, params, config} =
               Bumblebee.load_model({:hf, "microsoft/beit-base-patch16-224-pt22k"})

      assert %Bumblebee.Vision.Beit{architecture: :for_masked_image_modeling} = config

      input = Nx.broadcast(0.5, {1, 3, 224, 224})
      output = Axon.predict(model, params, input, compiler: EXLA)

      assert Nx.shape(output.logits) == {1, 196, 8192}

      assert_all_close(
        output.logits[[0, 0, 0..2]],
        Nx.tensor([0.4336, 1.2613, -6.8080]),
        atol: 1.0e-4
      )
    end

    # TODO: This one is quite complex
    # @tag :slow
    # test "semantic segmentation model" do
    #   assert {:ok, model, params, config} =
    #            Bumblebee.load_model({:hf, "microsoft/beit-base-finetuned-ade-640-640"})

    #   assert %Bumblebee.Vision.Beit{architecture: :for_semantic_segmentation} = config

    #   input = Nx.broadcast(0.5, {1, 3, 640, 640})
    #   output = Axon.predict(model, params, input, compiler: EXLA)

    #   assert Nx.shape(output.logits) == {1, 150, 160, 160}

    #   assert_all_close(
    #     output.logits[[0, 0, 0..4, 0..4]],
    #     Nx.tensor([
    #       [-2.3867, -0.4092, -0.4876, -0.5591, -0.4831],
    #       [-0.7347, 0.5419, 0.5512, 0.5975, 0.5525],
    #       [-0.5454, 0.6628, 0.6547, 0.6841, 0.7236],
    #       [-0.5991, 0.6530, 0.7027, 0.6715, 0.7251],
    #       [-0.6501, 0.5600, 0.5752, 0.5784, 0.6402]
    #     ]),
    #     atol: 1.0e-4
    #   )
    # end
  end
end
