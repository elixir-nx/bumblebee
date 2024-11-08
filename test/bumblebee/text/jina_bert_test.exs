defmodule Bumblebee.Text.JinaBertTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @tag slow: true
  test "jina-embeddings-v2-small-en" do
    repo = {:hf, "jinaai/jina-embeddings-v2-small-en"}

    {:ok, %{model: model, params: params, spec: _spec}} =
      Bumblebee.load_model(repo,
        params_filename: "model.safetensors",
        spec_overrides: [architecture: :base]
      )

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.all_close(
             outputs.hidden_state[[.., 1..3, 1..3]],
             Nx.tensor([
               [-0.1346, 0.1457, 0.5572],
               [-0.1383, 0.1412, 0.5643],
               [-0.1125, 0.1354, 0.5599]
             ])
           )
  end

  @tag :skip
  test ":base" do
    repo = {:hf, "doesnotexist/tiny-random-JinaBert"}

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(repo)

    assert %Bumblebee.Text.JinaBert{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.2331, 1.7817, 1.1736], [-1.1001, 1.3922, -0.3391], [0.0408, 0.8677, -0.0779]]
      ])
    )
  end

  @tag :skip
  test ":for_masked_language_modeling" do
    repo = {:hf, "doesnotexist/tiny-random-JinaBert"}

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(repo)

    assert %Bumblebee.Text.Bert{architecture: :for_masked_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1124}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([[[-0.0127, 0.0508, 0.0904], [0.1151, 0.1189, 0.0922], [0.0089, 0.1132, -0.2470]]])
    )
  end
end
