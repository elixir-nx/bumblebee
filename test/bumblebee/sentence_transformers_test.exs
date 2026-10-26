defmodule Bumblebee.SentenceTransformersTest do
  use ExUnit.Case, async: true

  alias Bumblebee.SentenceTransformers

  defp assert_all_close(a, b, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    diff = Nx.abs(Nx.subtract(a, b))
    max_diff = diff |> Nx.reduce_max() |> Nx.to_number()
    assert max_diff <= atol, "expected tensors to be close, got max diff #{max_diff} > #{atol}"
  end

  setup do
    model =
      "input_ids"
      |> Axon.input(shape: {nil, nil})
      |> Axon.nx(fn input_ids ->
        batch_size = Nx.axis_size(input_ids, 0)
        seq_len = Nx.axis_size(input_ids, 1)

        hidden_state =
          1.0
          |> Nx.broadcast({batch_size, seq_len, 4})
          |> Nx.as_type({:f, 32})

        %{hidden_state: hidden_state}
      end)

    params = %Axon.ModelState{data: %{}, parameters: %{}}
    model_info = %{model: model, params: params, spec: nil}

    [model_info: model_info]
  end

  describe "pipeline modules" do
    @tag :tmp_dir
    test "loads modules.json and builds pooling + dense + normalize pipeline", %{
      model_info: model_info,
      tmp_dir: tmp_dir
    } do
      modules = [
        %{
          "idx" => 0,
          "name" => "0",
          "path" => "",
          "type" => "sentence_transformers.models.Transformer"
        },
        %{
          "idx" => 1,
          "name" => "1",
          "path" => "1_Pooling",
          "type" => "sentence_transformers.models.Pooling"
        },
        %{
          "idx" => 2,
          "name" => "2",
          "path" => "2_Dense",
          "type" => "sentence_transformers.models.Dense"
        },
        %{
          "idx" => 3,
          "name" => "3",
          "path" => "3_Normalize",
          "type" => "sentence_transformers.models.Normalize"
        }
      ]

      File.write!(Path.join(tmp_dir, "modules.json"), Jason.encode!(modules))

      # 1_Pooling config
      File.mkdir_p!(Path.join(tmp_dir, "1_Pooling"))
      pooling_config = %{"pooling_mode_mean_tokens" => true}
      File.write!(Path.join(tmp_dir, "1_Pooling/config.json"), Jason.encode!(pooling_config))

      # 2_Dense config and weights
      File.mkdir_p!(Path.join(tmp_dir, "2_Dense"))

      dense_config = %{
        "in_features" => 4,
        "out_features" => 2,
        "bias" => true,
        "activation_function" => "torch.nn.modules.activation.Tanh"
      }

      File.write!(Path.join(tmp_dir, "2_Dense/config.json"), Jason.encode!(dense_config))

      weight = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      bias = Nx.tensor([0.1, -0.1])
      tensors = %{"linear.weight" => weight, "linear.bias" => bias}
      Safetensors.write!(Path.join(tmp_dir, "2_Dense/model.safetensors"), tensors)

      # 3_Normalize (no extra config file needed)
      File.mkdir_p!(Path.join(tmp_dir, "3_Normalize"))

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, tmp_dir}, model_info)

      {_init, predict} = Axon.build(new_model_info.model)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2]]),
        "attention_mask" => Nx.tensor([[1, 1]])
      }

      output = predict.(new_model_info.params, inputs)

      assert %{embedding: embedding, pooled_state: pooled, hidden_state: _} = output
      assert Nx.shape(embedding) == {1, 2}
      assert Nx.shape(pooled) == {1, 2}

      # Verify L2 normalization: norm should equal 1.0
      norm = embedding |> Nx.LinAlg.norm() |> Nx.to_number()
      assert_in_delta norm, 1.0, 1.0e-5
    end

    @tag :tmp_dir
    test "supports fuse_dense option to merge consecutive linear layers", %{
      model_info: model_info,
      tmp_dir: tmp_dir
    } do
      modules = [
        %{
          "idx" => 0,
          "name" => "0",
          "path" => "",
          "type" => "sentence_transformers.models.Transformer"
        },
        %{
          "idx" => 1,
          "name" => "1",
          "path" => "1_Dense",
          "type" => "sentence_transformers.models.Dense"
        },
        %{
          "idx" => 2,
          "name" => "2",
          "path" => "2_Dense",
          "type" => "sentence_transformers.models.Dense"
        }
      ]

      File.write!(Path.join(tmp_dir, "modules.json"), Jason.encode!(modules))

      # 1_Dense: 4 -> 8 (no bias, identity activation)
      File.mkdir_p!(Path.join(tmp_dir, "1_Dense"))

      c1 = %{
        "in_features" => 4,
        "out_features" => 8,
        "bias" => false,
        "activation_function" => nil
      }

      File.write!(Path.join(tmp_dir, "1_Dense/config.json"), Jason.encode!(c1))
      w1 = Nx.broadcast(0.5, {8, 4})

      Safetensors.write!(Path.join(tmp_dir, "1_Dense/model.safetensors"), %{"linear.weight" => w1})

      # 2_Dense: 8 -> 2 (no bias, identity activation)
      File.mkdir_p!(Path.join(tmp_dir, "2_Dense"))

      c2 = %{
        "in_features" => 8,
        "out_features" => 2,
        "bias" => false,
        "activation_function" => nil
      }

      File.write!(Path.join(tmp_dir, "2_Dense/config.json"), Jason.encode!(c2))
      w2 = Nx.broadcast(0.25, {2, 8})

      Safetensors.write!(Path.join(tmp_dir, "2_Dense/model.safetensors"), %{"linear.weight" => w2})

      assert {:ok, fused_info} =
               SentenceTransformers.load_embedding_head({:local, tmp_dir}, model_info,
                 fuse_dense: true
               )

      assert Map.has_key?(fused_info.params.data, "1_Dense_2_Dense")
      fused_kernel = fused_info.params.data["1_Dense_2_Dense"]["kernel"]
      assert Nx.shape(fused_kernel) == {4, 2}
    end

    @tag :tmp_dir
    test "does not fuse linear dense layers when bias is omitted (defaults to true)", %{
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

      # 2_Dense: 4 -> 8 (bias omitted, defaults to true)
      dense1_dir = Path.join(dir, "2_Dense")
      File.mkdir_p!(dense1_dir)

      File.write!(
        Path.join(dense1_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 4,
          "out_features" => 8,
          "activation_function" => "torch.nn.modules.linear.Identity"
        })
      )

      k1 = 0.5 |> Nx.broadcast({8, 4}) |> Nx.as_type({:f, 32})
      b1 = 0.1 |> Nx.broadcast({8}) |> Nx.as_type({:f, 32})

      Safetensors.write!(Path.join(dense1_dir, "model.safetensors"), %{
        "linear.weight" => k1,
        "linear.bias" => b1
      })

      # 3_Dense: 8 -> 4 (bias omitted, defaults to true)
      dense2_dir = Path.join(dir, "3_Dense")
      File.mkdir_p!(dense2_dir)

      File.write!(
        Path.join(dense2_dir, "config.json"),
        Jason.encode!(%{
          "in_features" => 8,
          "out_features" => 4,
          "activation_function" => "torch.nn.modules.linear.Identity"
        })
      )

      k2 = 0.25 |> Nx.broadcast({4, 8}) |> Nx.as_type({:f, 32})
      b2 = 0.2 |> Nx.broadcast({4}) |> Nx.as_type({:f, 32})

      Safetensors.write!(Path.join(dense2_dir, "model.safetensors"), %{
        "linear.weight" => k2,
        "linear.bias" => b2
      })

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info,
                 fuse_dense: true
               )

      assert Map.has_key?(new_model_info.params.data, "2_Dense")
      assert Map.has_key?(new_model_info.params.data, "3_Dense")
      refute Map.has_key?(new_model_info.params.data, "2_Dense_3_Dense")
    end

    @tag :tmp_dir
    test "loads max_tokens pooling", %{tmp_dir: dir} do
      custom_model =
        "input_ids"
        |> Axon.input(shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([[[1.0, 5.0, 2.0, 3.0], [4.0, 2.0, 6.0, 1.0], [99.0, 99.0, 99.0, 99.0]]])

          %{hidden_state: hidden_state}
        end)

      model_info = %{
        model: custom_model,
        params: %Axon.ModelState{data: %{}, parameters: %{}},
        spec: nil
      }

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

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert_all_close(output.embedding, Nx.tensor([[4.0, 5.0, 6.0, 3.0]]))
    end

    @tag :tmp_dir
    test "loads last_token pooling with both key variants", %{tmp_dir: dir} do
      custom_model =
        "input_ids"
        |> Axon.input(shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([[[1.0, 5.0, 2.0, 3.0], [4.0, 2.0, 6.0, 1.0], [99.0, 99.0, 99.0, 99.0]]])

          %{hidden_state: hidden_state}
        end)

      model_info = %{
        model: custom_model,
        params: %Axon.ModelState{data: %{}, parameters: %{}},
        spec: nil
      }

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

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert_all_close(output.embedding, Nx.tensor([[4.0, 2.0, 6.0, 1.0]]))
    end

    @tag :tmp_dir
    test "concatenates multiple pooling modes in PyTorch order", %{tmp_dir: dir} do
      custom_model =
        "input_ids"
        |> Axon.input(shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([[[1.0, 5.0, 2.0, 3.0], [4.0, 2.0, 6.0, 1.0], [99.0, 99.0, 99.0, 99.0]]])

          %{hidden_state: hidden_state}
        end)

      model_info = %{
        model: custom_model,
        params: %Axon.ModelState{data: %{}, parameters: %{}},
        spec: nil
      }

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => true,
        "pooling_mode_max_tokens" => true
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      # Max tokens: [4.0, 5.0, 6.0, 3.0], Mean tokens: [2.5, 3.5, 4.0, 2.0]
      expected = Nx.tensor([[4.0, 5.0, 6.0, 3.0, 2.5, 3.5, 4.0, 2.0]])
      assert_all_close(output.embedding, expected)
    end

    @tag :tmp_dir
    test "loads weightedmean_tokens pooling", %{tmp_dir: dir} do
      custom_model =
        "input_ids"
        |> Axon.input(shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state = Nx.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
          %{hidden_state: hidden_state}
        end)

      model_info = %{
        model: custom_model,
        params: %Axon.ModelState{data: %{}, parameters: %{}},
        spec: nil
      }

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_weightedmean_tokens" => true
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 0]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      # 1 * [1, 2] + 2 * [3, 4] = [7, 10]; sum weights = 3
      assert_all_close(output.embedding, Nx.tensor([[7.0 / 3, 10.0 / 3]]))
    end

    @tag :tmp_dir
    test "excludes prompt tokens from pooling when include_prompt is false", %{
      tmp_dir: dir
    } do
      custom_model =
        "input_ids"
        |> Axon.input(shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([
              [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]],
              [[5.0, 4.0], [6.0, 5.0], [7.0, 6.0], [8.0, 7.0]]
            ])

          %{hidden_state: hidden_state}
        end)

      model_info = %{
        model: custom_model,
        params: %Axon.ModelState{data: %{}, parameters: %{}},
        spec: nil
      }

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_mean_tokens" => true,
        "include_prompt" => false
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
        "prompt_length" => Nx.tensor([1, 2])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      # First sequence: exclude 1 prompt token -> mean of [[2,1], [3,2], [4,3]] = [3, 2]
      # Second sequence: exclude 2 prompt tokens -> mean of [[7,6], [8,7]] = [7.5, 6.5]
      assert_all_close(output.embedding, Nx.tensor([[3.0, 2.0], [7.5, 6.5]]))
    end

    @tag :tmp_dir
    test "excludes prompt tokens with left padding when include_prompt is false", %{
      tmp_dir: dir
    } do
      custom_model =
        "input_ids"
        |> Axon.input(shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          hidden_state =
            Nx.tensor([
              [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]]
            ])

          %{hidden_state: hidden_state}
        end)

      model_info = %{
        model: custom_model,
        params: %Axon.ModelState{data: %{}, parameters: %{}},
        spec: nil
      }

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      pooling_config = %{
        "pooling_mode_mean_tokens" => true,
        "include_prompt" => false
      }

      File.write!(Path.join(pooling_dir, "config.json"), Jason.encode!(pooling_config))

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask" => Nx.tensor([[0, 0, 1, 1, 1, 1]]),
        "prompt_length" => Nx.tensor([1])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      # 2 pad tokens + 1 prompt token excluded -> mean of [[2,1], [3,2], [4,3]] = [3, 2]
      assert_all_close(output.embedding, Nx.tensor([[3.0, 2.0]]))
    end

    @tag :tmp_dir
    test "loads LayerNorm module", %{model_info: model_info, tmp_dir: dir} do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"},
        %{"idx" => 2, "path" => "2_LayerNorm", "type" => "sentence_transformers.models.LayerNorm"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      File.write!(
        Path.join(pooling_dir, "config.json"),
        Jason.encode!(%{"pooling_mode_mean_tokens" => true})
      )

      ln_dir = Path.join(dir, "2_LayerNorm")
      File.mkdir_p!(ln_dir)

      File.write!(Path.join(ln_dir, "config.json"), Jason.encode!(%{"eps" => 1.0e-5}))

      tensors = %{
        "weight" => Nx.tensor([2.0, 2.0, 2.0, 2.0]),
        "bias" => Nx.tensor([1.0, 1.0, 1.0, 1.0])
      }

      Safetensors.write!(Path.join(ln_dir, "model.safetensors"), tensors)

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2]]),
        "attention_mask" => Nx.tensor([[1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert_all_close(output.embedding, Nx.tensor([[1.0, 1.0, 1.0, 1.0]]))
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
        "linear.weight" => 0.5 |> Nx.broadcast({4, 4}) |> Nx.as_type({:f, 32})
      }

      Safetensors.write!(Path.join(dense_dir, "model.safetensors"), weights)

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3]]),
        "attention_mask" => Nx.tensor([[1, 1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      expected = Nx.tensor([[0.96402758, 0.96402758, 0.96402758, 0.96402758]])
      assert_all_close(output.embedding, expected)
    end

    @tag :tmp_dir
    test "loads config metadata with fallback filenames", %{tmp_dir: dir} do
      # Test fallback to sentence_bert_config.json
      st_config = %{
        "prompts" => %{
          "query" => "Represent this query: ",
          "passage" => "Represent this passage: "
        },
        "default_prompt_name" => "query",
        "similarity_fn_name" => "cosine",
        "max_seq_length" => 256
      }

      File.write!(Path.join(dir, "sentence_bert_config.json"), Jason.encode!(st_config))

      assert {:ok, config} = SentenceTransformers.load_config({:local, dir})
      assert config.default_prompt_name == "query"
      assert config.similarity_fn_name == :cosine
      assert config.max_seq_length == 256
      assert config.prompts["query"] == "Represent this query: "
    end

    @tag :tmp_dir
    test "loads config metadata from 0_Transformer/sentence_bert_config.json fallback", %{
      tmp_dir: dir
    } do
      transformer_dir = Path.join(dir, "0_Transformer")
      File.mkdir_p!(transformer_dir)

      st_config = %{
        "prompts" => %{"query" => "Represent query from 0_Transformer: "},
        "default_prompt_name" => "query",
        "similarity_fn_name" => "dot"
      }

      File.write!(
        Path.join(transformer_dir, "sentence_bert_config.json"),
        Jason.encode!(st_config)
      )

      assert {:ok, config} = SentenceTransformers.load_config({:local, dir})
      assert config.default_prompt_name == "query"
      assert config.similarity_fn_name == :dot
      assert config.prompts["query"] == "Represent query from 0_Transformer: "
    end

    @tag :tmp_dir
    test "config priority: config_sentence_transformers.json > sentence_bert_config.json > 0_Transformer/sentence_bert_config.json",
         %{tmp_dir: dir} do
      transformer_dir = Path.join(dir, "0_Transformer")
      File.mkdir_p!(transformer_dir)

      File.write!(
        Path.join(transformer_dir, "sentence_bert_config.json"),
        Jason.encode!(%{"default_prompt_name" => "from_0_transformer"})
      )

      File.write!(
        Path.join(dir, "sentence_bert_config.json"),
        Jason.encode!(%{"default_prompt_name" => "from_root_legacy"})
      )

      # 1. sentence_bert_config.json takes precedence over 0_Transformer
      assert {:ok, config} = SentenceTransformers.load_config({:local, dir})
      assert config.default_prompt_name == "from_root_legacy"

      # 2. config_sentence_transformers.json takes precedence over both
      File.write!(
        Path.join(dir, "config_sentence_transformers.json"),
        Jason.encode!(%{"default_prompt_name" => "from_canonical"})
      )

      assert {:ok, config} = SentenceTransformers.load_config({:local, dir})
      assert config.default_prompt_name == "from_canonical"
    end

    test "causal fallback: infers 'last' pooling for Bumblebee causal specs" do
      causal_specs = [
        struct(Bumblebee.Text.Gemma),
        struct(Bumblebee.Text.Gemma3Text),
        struct(Bumblebee.Text.Llama),
        struct(Bumblebee.Text.Mistral),
        struct(Bumblebee.Text.Phi),
        struct(Bumblebee.Text.Phi3),
        struct(Bumblebee.Text.Gpt2),
        struct(Bumblebee.Text.GptBigCode),
        struct(Bumblebee.Text.GptNeoX),
        struct(Bumblebee.Text.Qwen3),
        struct(Bumblebee.Text.SmolLm3)
      ]

      for spec <- causal_specs do
        model_info = %{spec: spec}

        assert SentenceTransformers.infer_fallback_pooling_mode(
                 {:local, "nonexistent"},
                 model_info
               ) == "last"
      end
    end

    test "infers 'mean' pooling for Bumblebee non-causal specs" do
      non_causal_specs = [
        struct(Bumblebee.Text.Bert),
        struct(Bumblebee.Text.Roberta),
        struct(Bumblebee.Text.Distilbert),
        struct(Bumblebee.Text.Albert)
      ]

      for spec <- non_causal_specs do
        model_info = %{spec: spec}

        assert SentenceTransformers.infer_fallback_pooling_mode(
                 {:local, "nonexistent"},
                 model_info
               ) == "mean"
      end
    end

    @tag :tmp_dir
    test "infers 'last' pooling for config.json with *ForCausalLM or *LMHeadModel architectures",
         %{tmp_dir: dir} do
      causal_archs = [
        ["LlamaForCausalLM"],
        ["MistralForCausalLM"],
        ["GemmaForCausalLM"],
        ["PhiForCausalLM"],
        ["Qwen2ForCausalLM"],
        ["GPT2LMHeadModel"]
      ]

      for archs <- causal_archs do
        File.write!(Path.join(dir, "config.json"), Jason.encode!(%{"architectures" => archs}))

        assert SentenceTransformers.infer_fallback_pooling_mode({:local, dir}, %{spec: nil}) ==
                 "last"
      end
    end

    @tag :tmp_dir
    test "infers 'mean' pooling for non-causal architectures or when is_causal is false", %{
      tmp_dir: dir
    } do
      File.write!(
        Path.join(dir, "config.json"),
        Jason.encode!(%{"architectures" => ["BertModel"]})
      )

      assert SentenceTransformers.infer_fallback_pooling_mode({:local, dir}, %{spec: nil}) ==
               "mean"

      File.write!(
        Path.join(dir, "config.json"),
        Jason.encode!(%{
          "architectures" => ["LlamaForCausalLM"],
          "is_causal" => false
        })
      )

      assert SentenceTransformers.infer_fallback_pooling_mode({:local, dir}, %{spec: nil}) ==
               "mean"
    end

    @tag :tmp_dir
    test "returns error when modules.json is missing", %{model_info: model_info, tmp_dir: dir} do
      assert {:error, "could not find modules.json in the repository"} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)
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
              "expected the first module in modules.json to be Transformer or StaticEmbedding"} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)
    end

    @tag :tmp_dir
    test "returns error on unsupported module", %{model_info: model_info, tmp_dir: dir} do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Unknown", "type" => "sentence_transformers.models.Unknown"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      assert {:error, "unsupported SentenceTransformers module \"Unknown\""} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)
    end

    @tag :tmp_dir
    test "returns error on unsupported activation function", %{
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
          "activation_function" => "torch.nn.modules.activation.UnsupportedAct"
        })
      )

      assert {:error,
              "unsupported activation function \"torch.nn.modules.activation.UnsupportedAct\" in 2_Dense"} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)
    end

    @tag :tmp_dir
    test "loads dense layer with Sigmoid activation", %{model_info: model_info, tmp_dir: dir} do
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

      weights = %{
        "linear.weight" => 0.0 |> Nx.broadcast({4, 4}) |> Nx.as_type({:f, 32})
      }

      Safetensors.write!(Path.join(dense_dir, "model.safetensors"), weights)

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2]]),
        "attention_mask" => Nx.tensor([[1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      # sigmoid(0) = 0.5
      assert_all_close(output.embedding, Nx.tensor([[0.5, 0.5, 0.5, 0.5]]))
    end

    @tag :tmp_dir
    test "loads dense layer with use_residual: true", %{model_info: model_info, tmp_dir: dir} do
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
          "activation_function" => nil,
          "use_residual" => true
        })
      )

      weights = %{
        "linear.weight" => 0.5 |> Nx.broadcast({4, 4}) |> Nx.as_type({:f, 32})
      }

      Safetensors.write!(Path.join(dense_dir, "model.safetensors"), weights)

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2]]),
        "attention_mask" => Nx.tensor([[1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      # input pooled is [1, 1, 1, 1]. linear(x) = [2, 2, 2, 2]. with residual: [3, 3, 3, 3]
      assert_all_close(output.embedding, Nx.tensor([[3.0, 3.0, 3.0, 3.0]]))
    end

    @tag :tmp_dir
    test "supports truncate_dim and re-normalization", %{model_info: model_info, tmp_dir: dir} do
      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{"idx" => 1, "path" => "1_Pooling", "type" => "sentence_transformers.models.Pooling"},
        %{"idx" => 2, "path" => "2_Normalize", "type" => "sentence_transformers.models.Normalize"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      File.write!(
        Path.join(pooling_dir, "config.json"),
        Jason.encode!(%{"pooling_mode_mean_tokens" => true})
      )

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info,
                 truncate_dim: 2
               )

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2]]),
        "attention_mask" => Nx.tensor([[1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      assert Nx.shape(output.embedding) == {1, 2}
      # Re-normalized L2 norm must be 1.0
      norm = output.embedding |> Nx.LinAlg.norm() |> Nx.to_number()
      assert_in_delta norm, 1.0, 1.0e-5
    end

    test "truncate_embeddings helper function" do
      emb = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      truncated = SentenceTransformers.truncate_embeddings(emb, 2)
      assert Nx.shape(truncated) == {1, 2}
      assert_all_close(truncated, Nx.tensor([[1.0, 1.0]]))

      truncated_norm = SentenceTransformers.truncate_embeddings(emb, 2, normalize: true)
      norm = truncated_norm |> Nx.LinAlg.norm() |> Nx.to_number()
      assert_in_delta norm, 1.0, 1.0e-5
    end

    @tag :tmp_dir
    test "loads WeightedLayerPooling module", %{tmp_dir: dir} do
      custom_model =
        "input_ids"
        |> Axon.input(shape: {nil, nil})
        |> Axon.nx(fn _input_ids ->
          l1 = Nx.broadcast(1.0, {1, 2, 4})
          l2 = Nx.broadcast(2.0, {1, 2, 4})
          l3 = Nx.broadcast(3.0, {1, 2, 4})
          stacked = Nx.stack([l1, l2, l3], axis: 0)
          %{hidden_state: stacked}
        end)

      model_info = %{
        model: custom_model,
        params: %Axon.ModelState{data: %{}, parameters: %{}},
        spec: nil
      }

      modules = [
        %{"idx" => 0, "path" => "", "type" => "sentence_transformers.models.Transformer"},
        %{
          "idx" => 1,
          "path" => "1_WeightedLayerPooling",
          "type" => "sentence_transformers.models.WeightedLayerPooling"
        },
        %{"idx" => 2, "path" => "2_Pooling", "type" => "sentence_transformers.models.Pooling"}
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      wlp_dir = Path.join(dir, "1_WeightedLayerPooling")
      File.mkdir_p!(wlp_dir)

      File.write!(
        Path.join(wlp_dir, "config.json"),
        Jason.encode!(%{"layer_start" => 0, "num_hidden_layers" => 2})
      )

      weights = %{
        "layer_weights" => Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})
      }

      Safetensors.write!(Path.join(wlp_dir, "model.safetensors"), weights)

      pooling_dir = Path.join(dir, "2_Pooling")
      File.mkdir_p!(pooling_dir)

      File.write!(
        Path.join(pooling_dir, "config.json"),
        Jason.encode!(%{"pooling_mode_mean_tokens" => true})
      )

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2]]),
        "attention_mask" => Nx.tensor([[1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(new_model_info.model)
      output = predict_fn.(new_model_info.params, inputs)

      expected = Nx.broadcast(2.3333333, {1, 4})
      assert_all_close(output.embedding, expected)
    end

    @tag :tmp_dir
    test "loads StaticEmbedding model (Model2Vec style)", %{tmp_dir: dir} do
      modules = [
        %{
          "idx" => 0,
          "name" => "0",
          "path" => "0_StaticEmbedding",
          "type" => "sentence_transformers.models.StaticEmbedding"
        },
        %{
          "idx" => 1,
          "name" => "1",
          "path" => "1_Pooling",
          "type" => "sentence_transformers.models.Pooling"
        }
      ]

      File.write!(Path.join(dir, "modules.json"), Jason.encode!(modules))

      static_dir = Path.join(dir, "0_StaticEmbedding")
      File.mkdir_p!(static_dir)

      emb_matrix = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], type: {:f, 32})

      Safetensors.write!(Path.join(static_dir, "model.safetensors"), %{
        "embedding.weight" => emb_matrix
      })

      pooling_dir = Path.join(dir, "1_Pooling")
      File.mkdir_p!(pooling_dir)

      File.write!(
        Path.join(pooling_dir, "config.json"),
        Jason.encode!(%{"pooling_mode_mean_tokens" => true})
      )

      assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})

      inputs = %{
        "input_ids" => Nx.tensor([[0, 2]]),
        "attention_mask" => Nx.tensor([[1, 1]])
      }

      {_init_fn, predict_fn} = Axon.build(model_info.model)
      output = predict_fn.(model_info.params, inputs)

      assert_all_close(output.embedding, Nx.tensor([[3.0, 4.0]]))
    end

    @tag :tmp_dir
    test "updates ModelState.parameters for added layers", %{tmp_dir: dir} do
      base_model =
        "input_ids"
        |> Axon.input(shape: {nil, 4})
        |> Axon.nx(fn x -> %{hidden_state: Nx.new_axis(x, 1)} end)

      base_params = %Axon.ModelState{
        data: %{"base" => %{"kernel" => Nx.broadcast(1.0, {4, 4})}},
        parameters: %{"base" => ["kernel"]},
        frozen_parameters: %{},
        state: %{}
      }

      model_info = %{model: base_model, params: base_params, spec: nil}

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

      weights = %{
        "linear.weight" => 0.5 |> Nx.broadcast({8, 4}) |> Nx.as_type({:f, 32}),
        "linear.bias" => 0.1 |> Nx.broadcast({8}) |> Nx.as_type({:f, 32})
      }

      Safetensors.write!(Path.join(dense_dir, "model.safetensors"), weights)

      assert {:ok, new_model_info} =
               SentenceTransformers.load_embedding_head({:local, dir}, model_info)

      assert Enum.sort(new_model_info.params.parameters["2_Dense"]) == ["bias", "kernel"]
      assert new_model_info.params.parameters["base"] == ["kernel"]
    end
  end

  @tag :slow
  @tag :network
  test "works directly with Bumblebee.Text.text_embedding and piping" do
    repo = {:hf, "sentence-transformers/all-MiniLM-L6-v2"}

    # Piping: Bumblebee.load_model |> SentenceTransformers.load_embedding_head
    assert {:ok, model_info} =
             repo
             |> Bumblebee.load_model(architecture: :base)
             |> SentenceTransformers.load_embedding_head(repo)

    assert {:ok, tokenizer} = Bumblebee.load_tokenizer(repo)

    # Uses standard Bumblebee serving directly!
    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)
    result = Nx.Serving.run(serving, "Hello world")

    assert %{embedding: embedding} = result
    assert Nx.shape(embedding) == {384}

    first_5 = Nx.to_flat_list(embedding[0..4])
    py_reference = [-0.03447720, 0.03102321, 0.00673497, 0.02610898, -0.03936199]

    first_5
    |> Enum.zip(py_reference)
    |> Enum.each(fn {val, ref} ->
      assert_in_delta val, ref, 1.0e-5
    end)
  end

  @tag :slow
  @tag :network
  test "end-to-end integration test with sentence-transformers/all-MiniLM-L6-v2" do
    repo = {:hf, "sentence-transformers/all-MiniLM-L6-v2"}

    assert {:ok, model_info} = SentenceTransformers.load_model(repo)
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer(repo)

    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)
    result = Nx.Serving.run(serving, "Hello world")

    assert %{embedding: embedding} = result
    assert Nx.shape(embedding) == {384}

    # Reference values from PyTorch sentence-transformers 6.1.0 for "Hello world":
    # [-0.034477, 0.031023, 0.006735, 0.026109, -0.039362]
    first_5 = Nx.to_flat_list(embedding[0..4])
    py_reference = [-0.03447720, 0.03102321, 0.00673497, 0.02610898, -0.03936199]

    first_5
    |> Enum.zip(py_reference)
    |> Enum.each(fn {val, ref} ->
      assert_in_delta val, ref, 1.0e-5
    end)
  end

  @tag :slow
  @tag :network
  test "end-to-end integration test with intfloat/e5-small-v2" do
    repo = {:hf, "intfloat/e5-small-v2"}

    assert {:ok, model_info} = SentenceTransformers.load_model(repo)
    assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer(repo)

    serving = SentenceTransformers.text_embedding(model_info, tokenizer)
    result = Nx.Serving.run(serving, "query: Cats are cute.")

    assert %{embedding: embedding} = result
    assert Nx.shape(embedding) == {384}

    # Reference values from PyTorch sentence-transformers 6.1.0 for "query: Cats are cute.":
    # [-0.010973, 0.073814, 0.011410]
    slice = Nx.to_flat_list(embedding[1..3])
    py_reference = [-0.01097320, 0.07381421, 0.01141022]

    slice
    |> Enum.zip(py_reference)
    |> Enum.each(fn {val, ref} ->
      assert_in_delta val, ref, 1.0e-5
    end)
  end
end
