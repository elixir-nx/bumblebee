defmodule Bumblebee.HuggingFace.SentenceTransformersTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  setup do
    model =
      Axon.input("input_ids", shape: {nil, nil})
      |> Axon.nx(fn input_ids ->
        # Mock base model output with hidden_state {batch_size, seq_len, 4}
        batch_size = Nx.axis_size(input_ids, 0)
        seq_len = Nx.axis_size(input_ids, 1)

        hidden_state =
          Nx.broadcast(1.0, {batch_size, seq_len, 4})
          |> Nx.as_type({:f, 32})

        %{hidden_state: hidden_state}
      end)

    params = Axon.ModelState.empty()
    spec = Bumblebee.configure(Bumblebee.Text.Bert, architecture: :base)

    model_info = %{model: model, params: params, spec: spec}

    [model_info: model_info]
  end

  describe "load_embedding_head/3" do
    @tag :tmp_dir
    test "loads pooling and normalize head", %{model_info: model_info, tmp_dir: dir} do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"},
        %{"idx" => 2, "path" => "2_Normalize", "type" => "sentence_transformers.models.Normalize"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "word_embedding_dimension" => 4,
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => true,
        "pooling_mode_max_tokens" => false
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert Map.has_key?(output, :embedding)
      assert Map.has_key?(output, :hidden_state)
      assert Nx.shape(output.embedding) == {1, 4}

      # Normalization check: L2 norm of output embedding must be 1.0
      norm = Nx.LinAlg.norm(output.embedding, axes: [-1])
      assert_all_close(norm, Nx.tensor([1.0]), atol: 1.0e-5)
    end

    @tag :tmp_dir
    test "loads dense projection layers with parameters", %{model_info: model_info, tmp_dir: dir} do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"},
        %{"idx" => 2, "path" => "2_Dense", "type" => "sentence_transformers.models.Dense"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      File.write!(
        Path.join(pooling_dir, "config.json"),
        Jason.encode!(%{"pooling_mode_mean_tokens" => true})
      )

      dense_dir = Path.join(dir, "2_Dense")
      File.mkdir_p!(dense_dir)

      dense_config = %{
        "in_features" => 4,
        "out_features" => 8,
        "bias" => true,
        "activation_function" => "torch.nn.modules.linear.Identity"
      }

      File.write!(Path.join(dense_dir, "config.json"), Jason.encode!(dense_config))

      # Weight in PyTorch format: {out_features, in_features} = {8, 4}
      weights = %{
        "linear.weight" => Nx.broadcast(0.5, {8, 4}) |> Nx.as_type({:f, 32}),
        "linear.bias" => Nx.broadcast(0.1, {8}) |> Nx.as_type({:f, 32})
      }

      Safetensors.write!(Path.join(dense_dir, "model.safetensors"), weights)

      assert {:ok, new_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      assert Map.has_key?(new_model_info.params.data, "2_Dense")
      assert Nx.shape(new_model_info.params.data["2_Dense"]["kernel"]) == {4, 8}
      assert Nx.shape(new_model_info.params.data["2_Dense"]["bias"]) == {8}

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert Nx.shape(output.embedding) == {1, 8}
    end

    @tag :tmp_dir
    test "returns error when modules.json is missing", %{model_info: model_info, tmp_dir: dir} do
      assert {:error, "could not find modules.json in the repository"} =
               Bumblebee.load_embedding_head({:local, dir}, model_info)
    end

    @tag :tmp_dir
    test "returns error when first module is not Transformer", %{
      model_info: model_info,
      tmp_dir: dir
    } do
      modules = [
        %{"idx" => 0, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      assert {:error,
              "expected the first module in modules.json to be sentence_transformers.models.Transformer"} =
               Bumblebee.load_embedding_head({:local, dir}, model_info)
    end

    @tag :tmp_dir
    test "returns error on unsupported module", %{model_info: model_info, tmp_dir: dir} do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Unknown", "type" => "sentence_transformers.models.Unknown"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      assert {:error,
              "unsupported SentenceTransformers module \"sentence_transformers.models.Unknown\""} =
               Bumblebee.load_embedding_head({:local, dir}, model_info)
    end
  end

  @tag :slow
  test "end-to-end with unsloth/embeddinggemma-300m matches reference SentenceTransformers output" do
    assert {:ok, model_info} = Bumblebee.load_model({:hf, "unsloth/embeddinggemma-300m"})

    assert {:ok, model_info} =
             Bumblebee.load_embedding_head({:hf, "unsloth/embeddinggemma-300m"}, model_info)

    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/embeddinggemma-300m"})

    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)
    res = Nx.Serving.run(serving, "Hello world")

    assert Nx.shape(res.embedding) == {768}

    assert_all_close(
      res.embedding[0..4],
      Nx.tensor([-0.203079, 0.034759, 0.060166, -0.016863, 0.006666]),
      atol: 1.0e-4
    )
  end

  @tag :slow
  test "end-to-end with sentence-transformers/all-MiniLM-L6-v2" do
    assert {:ok, model_info} =
             Bumblebee.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})

    assert {:ok, model_info} =
             Bumblebee.load_embedding_head(
               {:hf, "sentence-transformers/all-MiniLM-L6-v2"},
               model_info
             )

    assert {:ok, tokenizer} =
             Bumblebee.load_tokenizer({:hf, "sentence-transformers/all-MiniLM-L6-v2"})

    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)
    res = Nx.Serving.run(serving, "Hello world")

    assert Nx.shape(res.embedding) == {384}

    norm = Nx.LinAlg.norm(res.embedding)
    assert_all_close(norm, Nx.tensor(1.0), atol: 1.0e-5)
  end
end
