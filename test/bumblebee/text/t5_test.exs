defmodule Bumblebee.Text.T5Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-T5Model"})

    assert %Bumblebee.Text.T5{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 8, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]] |> Nx.multiply(100),
      Nx.tensor([
        [[-0.0353, -0.2614, -0.0219], [0.0829, 0.0845, -0.1971], [-0.0208, -0.0795, -0.0401]]
      ])
    )

    assert_all_close(Nx.sum(outputs.hidden_state), -0.0235)
  end

  test ":base with gated feed-forward activation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-T5Model-feed_forward_proj-gated"}
             )

    assert %Bumblebee.Text.T5{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 8, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]] |> Nx.multiply(100),
      Nx.tensor([
        [[-0.0353, -0.2614, -0.0219], [0.0829, 0.0845, -0.1971], [-0.0208, -0.0795, -0.0401]]
      ])
    )

    assert_all_close(Nx.sum(outputs.hidden_state), -0.0235)
  end

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-T5ForConditionalGeneration"}
             )

    assert %Bumblebee.Text.T5{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 32100}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]] |> Nx.multiply(10_000),
      Nx.tensor([[[-0.0158, 0.0067, 0.0636], [0.0128, 0.0742, -0.0398], [0.0050, 0.0554, 0.0083]]])
    )
  end

  test ":for_conditional_generation without tied embeddings" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf,
                "bumblebee-testing/tiny-random-T5ForConditionalGeneration-tie_word_embeddings-False"}
             )

    assert %Bumblebee.Text.T5{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 32100}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]] |> Nx.multiply(10_000),
      Nx.tensor([
        [[0.0537, -0.0358, -0.2016], [0.0580, 0.2900, -0.0393], [0.0194, 0.0153, -0.0144]]
      ])
    )
  end

  test ":encoder" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-T5Model"},
               architecture: :encoder
             )

    assert %Bumblebee.Text.T5{architecture: :encoder} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0034, -0.0005, -0.0036], [-0.0002, 0.0029, 0.0021], [-0.0011, -0.0004, -0.0034]]
      ])
    )
  end
end
