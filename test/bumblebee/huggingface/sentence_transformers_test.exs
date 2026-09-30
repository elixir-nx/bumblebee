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

    [model_info: model_info, spec: spec]
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
    test "fuses consecutive linear dense layers when fuse_dense: true", %{
      model_info: model_info,
      tmp_dir: dir
    } do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"},
        %{"idx" => 2, "path" => "2_Dense", "type" => "sentence_transformers.models.Dense"},
        %{"idx" => 3, "path" => "3_Dense", "type" => "sentence_transformers.models.Dense"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      File.write!(
        Path.join(pooling_dir, "config.json"),
        Jason.encode!(%{"pooling_mode_mean_tokens" => true})
      )

      # 2_Dense: 4 -> 8
      dense1_dir = Path.join(dir, "2_Dense")
      File.mkdir_p!(dense1_dir)

      File.write!(
        Path.join(dense1_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 4,
          "out_features" => 8,
          "bias" => false,
          "activation_function" => "torch.nn.modules.linear.Identity"
        })
      )

      k1 = Nx.broadcast(0.5, {8, 4}) |> Nx.as_type({:f, 32})
      Safetensors.write!(Path.join(dense1_dir, "model.safetensors"), %{"linear.weight" => k1})

      # 3_Dense: 8 -> 4
      dense2_dir = Path.join(dir, "3_Dense")
      File.mkdir_p!(dense2_dir)

      File.write!(
        Path.join(dense2_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 8,
          "out_features" => 4,
          "bias" => false,
          "activation_function" => "torch.nn.modules.linear.Identity"
        })
      )

      k2 = Nx.broadcast(0.25, {4, 8}) |> Nx.as_type({:f, 32})
      Safetensors.write!(Path.join(dense2_dir, "model.safetensors"), %{"linear.weight" => k2})

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 1]])
      }

      # Separate (fuse_dense: false)
      assert {:ok, sep_model_info} =
               Bumblebee.load_embedding_head({:local, dir}, model_info, fuse_dense: false)

      assert Map.has_key?(sep_model_info.params.data, "2_Dense")
      assert Map.has_key?(sep_model_info.params.data, "3_Dense")

      {_, predict_sep} = Axon.build(sep_model_info.model)
      out_sep = predict_sep.(sep_model_info.params, inputs)

      # Fused (fuse_dense: true)
      assert {:ok, fus_model_info} =
               Bumblebee.load_embedding_head({:local, dir}, model_info, fuse_dense: true)

      assert Map.has_key?(fus_model_info.params.data, "2_Dense_3_Dense")
      assert Nx.shape(fus_model_info.params.data["2_Dense_3_Dense"]["kernel"]) == {4, 4}

      {_, predict_fus} = Axon.build(fus_model_info.model)
      out_fus = predict_fus.(fus_model_info.params, inputs)

      assert_all_close(out_sep.embedding, out_fus.embedding)
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

    @tag :tmp_dir
    test "loads max_tokens pooling", %{spec: spec, tmp_dir: dir} do
      custom_model =
        Axon.input("input_ids", shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([[[1.0, 5.0, 2.0, 3.0], [4.0, 2.0, 6.0, 1.0], [99.0, 99.0, 99.0, 99.0]]])

          %{hidden_state: hidden_state}
        end)

      model_info = %{model: custom_model, params: Axon.ModelState.empty(), spec: spec}

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => false,
        "pooling_mode_max_tokens" => true
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert_all_close(output.embedding, Nx.tensor([[4.0, 5.0, 6.0, 3.0]]))
    end

    @tag :tmp_dir
    test "loads last_token pooling", %{spec: spec, tmp_dir: dir} do
      custom_model =
        Axon.input("input_ids", shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([[[1.0, 5.0, 2.0, 3.0], [4.0, 2.0, 6.0, 1.0], [99.0, 99.0, 99.0, 99.0]]])

          %{hidden_state: hidden_state}
        end)

      model_info = %{model: custom_model, params: Axon.ModelState.empty(), spec: spec}

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => false,
        "pooling_mode_lasttoken" => true
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert_all_close(output.embedding, Nx.tensor([[4.0, 2.0, 6.0, 1.0]]))
    end

    @tag :tmp_dir
    test "loads mean_sqrt_len_tokens pooling", %{spec: spec, tmp_dir: dir} do
      custom_model =
        Axon.input("input_ids", shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([[[1.0, 5.0, 2.0, 3.0], [4.0, 2.0, 6.0, 1.0], [99.0, 99.0, 99.0, 99.0]]])

          %{hidden_state: hidden_state}
        end)

      model_info = %{model: custom_model, params: Axon.ModelState.empty(), spec: spec}

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => false,
        "pooling_mode_mean_sqrt_len_tokens" => true
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      expected = Nx.divide(Nx.tensor([[5.0, 7.0, 8.0, 4.0]]), :math.sqrt(2))
      assert_all_close(output.embedding, expected)
    end

    @tag :tmp_dir
    test "returns error when no supported pooling mode is enabled", %{
      model_info: model_info,
      tmp_dir: dir
    } do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => false
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:error, "no supported pooling mode found in 1_Pooling"} =
               Bumblebee.load_embedding_head({:local, dir}, model_info)
    end

    @tag :tmp_dir
    test "loads dense layer with activation (Tanh and ReLU)", %{
      model_info: model_info,
      tmp_dir: dir
    } do
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

      # 1. Tanh
      File.write!(
        Path.join(dense_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 4,
          "out_features" => 4,
          "bias" => false,
          "activation_function" => "torch.nn.modules.activation.Tanh"
        })
      )

      weights = %{
        "linear.weight" => Nx.broadcast(0.5, {4, 4}) |> Nx.as_type({:f, 32})
      }

      Safetensors.write!(Path.join(dense_dir, "model.safetensors"), weights)

      assert {:ok, new_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      expected = Nx.tensor([[0.96402758, 0.96402758, 0.96402758, 0.96402758]])
      assert_all_close(output.embedding, expected)

      # 2. ReLU with negative weights
      File.write!(
        Path.join(dense_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 4,
          "out_features" => 4,
          "bias" => false,
          "activation_function" => "torch.nn.modules.activation.ReLU"
        })
      )

      neg_weights = %{
        "linear.weight" => Nx.broadcast(-0.5, {4, 4}) |> Nx.as_type({:f, 32})
      }

      Safetensors.write!(Path.join(dense_dir, "model.safetensors"), neg_weights)

      assert {:ok, relu_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      {_init_fn, predict_relu} = Axon.build(relu_model_info.model)
      output_relu = predict_relu.(relu_model_info.params, inputs)

      assert_all_close(output_relu.embedding, Nx.broadcast(0.0, {1, 4}))
    end

    @tag :tmp_dir
    test "returns error on unsupported activation function in dense config", %{
      model_info: model_info,
      tmp_dir: dir
    } do
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

      File.write!(
        Path.join(dense_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 4,
          "out_features" => 4,
          "bias" => false,
          "activation_function" => "torch.nn.modules.activation.Sigmoid"
        })
      )

      assert {:error,
              "unsupported activation function \"torch.nn.modules.activation.Sigmoid\" in 2_Dense"} =
               Bumblebee.load_embedding_head({:local, dir}, model_info)
    end

    @tag :tmp_dir
    test "handles Dropout module as pass-through", %{model_info: model_info, tmp_dir: dir} do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"},
        %{"idx" => 2, "path" => "2_Dropout", "type" => "sentence_transformers.models.Dropout"},
        %{"idx" => 3, "path" => "3_Normalize", "type" => "sentence_transformers.models.Normalize"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      File.write!(
        Path.join(pooling_dir, "config.json"),
        Jason.encode!(%{"pooling_mode_mean_tokens" => true})
      )

      assert {:ok, new_model_info} = Bumblebee.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert Nx.shape(output.embedding) == {1, 4}
      norm = Nx.LinAlg.norm(output.embedding, axes: [-1])
      assert_all_close(norm, Nx.tensor([1.0]), atol: 1.0e-5)
    end

    @tag :tmp_dir
    test "loads dense layer parameters from pytorch_model.bin", %{
      model_info: model_info,
      tmp_dir: dir
    } do
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

      File.write!(
        Path.join(dense_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 4,
          "out_features" => 8,
          "bias" => true,
          "activation_function" => "torch.nn.modules.linear.Identity"
        })
      )

      pkl =
        <<128, 2, "ccollections\nOrderedDict\nq", 0, ")Rq", 1, "(X", 13, 0, 0, 0,
          "linear.weightq", 2, "ctorch._utils\n_rebuild_tensor_v2\nq", 3, "((X", 7, 0, 0, 0,
          "storageq", 4, "ctorch\nFloatStorage\nq", 5, "X", 1, 0, 0, 0, "0q", 6, "X", 3, 0, 0, 0,
          "cpuq", 7, "K ", 116, 113, 8, "QK", 0, "(K", 8, "K", 4, 116, 113, 9, "(K", 4, "K", 1,
          116, 113, 10, 137, 104, 0, ")Rq", 11, 116, 113, 12, "Rq", 13, "X", 11, 0, 0, 0,
          "linear.biasq", 14, "h", 3, "((h", 4, "h", 5, "X", 1, 0, 0, 0, "1q", 15, "h", 7, "K", 8,
          116, 113, 16, "QK", 0, "K", 8, 133, 113, 17, "K", 1, 133, 113, 18, 137, 104, 0, ")Rq",
          19, 116, 113, 20, "Rq", 21, "u.">>

      data0 = :binary.copy(<<0, 0, 0, 63>>, 32)
      data1 = :binary.copy(<<205, 204, 204, 61>>, 8)

      files = [
        {~c"archive/data.pkl", pkl},
        {~c"archive/data/0", data0},
        {~c"archive/data/1", data1},
        {~c"archive/version", "3\n"}
      ]

      pt_path = Path.join(dense_dir, "pytorch_model.bin")
      {:ok, _} = :zip.create(to_charlist(pt_path), files)

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
  end

  @tag :slow
  test "end-to-end with unsloth/embeddinggemma-300m matches reference SentenceTransformers output" do
    assert {:ok, base_model} = Bumblebee.load_model({:hf, "unsloth/embeddinggemma-300m"})
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/embeddinggemma-300m"})

    # 1. Standard Separate Head
    assert {:ok, model_info} =
             Bumblebee.load_embedding_head({:hf, "unsloth/embeddinggemma-300m"}, base_model)

    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)
    res = Nx.Serving.run(serving, "Hello world")

    assert Nx.shape(res.embedding) == {768}

    assert_all_close(
      res.embedding[0..4],
      Nx.tensor([-0.203079, 0.034759, 0.060166, -0.016863, 0.006666]),
      atol: 1.0e-4
    )

    # 2. Optimized Fused Head
    assert {:ok, fused_model_info} =
             Bumblebee.load_embedding_head(
               {:hf, "unsloth/embeddinggemma-300m"},
               base_model,
               fuse_dense: true
             )

    fused_serving = Bumblebee.Text.text_embedding(fused_model_info, tokenizer)
    fused_res = Nx.Serving.run(fused_serving, "Hello world")

    assert_all_close(res.embedding, fused_res.embedding)
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

  @tag :slow
  test "end-to-end with sentence-transformers/distiluse-base-multilingual-cased-v1 (Dense with Tanh)" do
    assert {:ok, model_info} =
             Bumblebee.load_model(
               {:hf, "sentence-transformers/distiluse-base-multilingual-cased-v1"}
             )

    assert {:ok, model_info} =
             Bumblebee.load_embedding_head(
               {:hf, "sentence-transformers/distiluse-base-multilingual-cased-v1"},
               model_info
             )

    assert {:ok, tokenizer} =
             Bumblebee.load_tokenizer(
               {:hf, "sentence-transformers/distiluse-base-multilingual-cased-v1"}
             )

    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)
    res = Nx.Serving.run(serving, "Hello world")

    assert Nx.shape(res.embedding) == {512}

    assert_all_close(
      res.embedding[0..4],
      Nx.tensor([0.037437, 0.000251, -0.044534, -0.006018, -0.004641]),
      atol: 1.0e-4
    )
  end

  @tag :slow
  test "end-to-end with sentence-transformers/sentence-t5-base (T5 encoder with Dense projection)" do
    assert {:ok, model_info} =
             Bumblebee.load_model({:hf, "sentence-transformers/sentence-t5-base"})

    assert {:ok, model_info} =
             Bumblebee.load_embedding_head(
               {:hf, "sentence-transformers/sentence-t5-base"},
               model_info
             )

    assert {:ok, tokenizer} =
             Bumblebee.load_tokenizer({:hf, "sentence-transformers/sentence-t5-base"})

    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)
    res = Nx.Serving.run(serving, "Hello world")

    assert Nx.shape(res.embedding) == {768}

    norm = Nx.LinAlg.norm(res.embedding)
    assert_all_close(norm, Nx.tensor(1.0), atol: 1.0e-5)

    assert_all_close(
      res.embedding[0..4],
      Nx.tensor([0.001551, -0.056042, 0.026304, 0.058385, 0.007024]),
      atol: 1.0e-4
    )
  end
end
