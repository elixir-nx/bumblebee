defmodule Bumblebee.SentenceTransformers.PythonComparisonTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  alias Bumblebee.SentenceTransformers
  alias Bumblebee.SentenceTransformers.Pipeline
  alias Bumblebee.SentenceTransformers.TestSupport.PythonBridge

  @moduletag :slow
  @moduletag :python

  test "saved short token limit matches Python with root config, prompts and compilation" do
    text = Enum.join(List.duplicate("hello world", 12), " ")

    for prompt <- [nil, "Represent the query: "] do
      assert {:ok, %{dir: dir, python_output: expected}} =
               PythonBridge.generate_random_sentence_transformer(
                 max_seq_length: 8,
                 text: text,
                 prompt: prompt
               )

      try do
        assert {:ok, info} = SentenceTransformers.load_model({:local, dir})
        assert info.sentence_transformers.max_seq_length == 8
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        for opts <- [[], [compile: [batch_size: 1, sequence_length: 32]]] do
          actual =
            Nx.Serving.run(SentenceTransformers.text_embedding(info, tokenizer, opts), text).embedding

          assert_all_close(actual, Nx.squeeze(expected), atol: 1.0e-4)
        end
      after
        File.rm_rf!(dir)
      end
    end
  end

  test "pooling matches Python with left padding and fully masked sequences" do
    hidden = {3, 5, 4} |> Nx.iota(type: :f32) |> Nx.add(1)
    mask = Nx.tensor([[0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
    modes = ["mean", "cls", "max", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"]

    code = """
    import json
    import torch
    from sentence_transformers.models import Pooling
    hidden = torch.tensor(#{Jason.encode!(Nx.to_list(hidden))})
    mask = torch.tensor(#{Jason.encode!(Nx.to_list(mask))})
    outputs = {}
    for mode in #{inspect(modes)}:
        result = Pooling(4, pooling_mode=mode)({"token_embeddings": hidden.clone(), "attention_mask": mask.clone()})["sentence_embedding"]
        outputs[mode] = {"values": torch.nan_to_num(result, neginf=0).tolist(), "negative_infinity": torch.isneginf(result).int().tolist()}
    print("__RESULT__:" + json.dumps(outputs))
    """

    expected = PythonBridge.run_python_code(code, ["sentence-transformers", "torch"], type: :raw)

    for mode <- modes do
      assert {:ok, model} =
               Pipeline.pooling_layer(
                 Axon.input("hidden"),
                 Axon.input("mask"),
                 %{"pooling_mode" => mode},
                 "pool"
               )

      {init, predict} = Axon.build(model)
      inputs = %{"hidden" => hidden, "mask" => mask}
      params = init.(inputs, Axon.ModelState.empty())
      actual = predict.(params, inputs)
      infinite = Nx.is_infinity(actual)
      assert Nx.to_list(infinite) == expected[mode]["negative_infinity"]

      assert_all_close(Nx.select(infinite, 0, actual), Nx.tensor(expected[mode]["values"]),
        atol: 1.0e-4
      )
    end
  end

  test "runtime prompts override saved defaults and work with compiled prompt exclusion" do
    assert {:ok, %{dir: dir}} =
             PythonBridge.generate_random_sentence_transformer(
               prompt: "Represent the query: ",
               include_prompt: false
             )

    try do
      assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
      assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

      for {prompt, compile} <- [
            {"Represent the document: ", nil},
            {"Represent the document: ", [batch_size: 2, sequence_length: 32]},
            {"", nil}
          ] do
        opts = [prompt: prompt]
        opts = if compile, do: Keyword.put(opts, :compile, compile), else: opts

        actual =
          model_info
          |> SentenceTransformers.text_embedding(tokenizer, opts)
          |> Nx.Serving.run("Hello world")

        expected = PythonBridge.run_sentence_transformers(dir, "Hello world", prompt: prompt)
        assert_all_close(actual.embedding, expected, atol: 1.0e-4)
      end
    after
      File.rm_rf!(dir)
    end
  end

  test "query and corpus helpers fall back when no named prompts are saved" do
    assert {:ok, %{dir: dir, python_output: expected}} =
             PythonBridge.generate_random_sentence_transformer()

    try do
      assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
      assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

      for helper <- [:encode_queries, :encode_corpus] do
        [actual] = apply(SentenceTransformers, helper, [model_info, tokenizer, ["Hello world"]])
        assert_all_close(actual.embedding, expected, atol: 1.0e-4)
      end
    after
      File.rm_rf!(dir)
    end
  end

  describe "On-the-fly generated random SentenceTransformers comparison" do
    @pooling_modes [
      "mean_tokens",
      "cls_token",
      "max_tokens",
      "mean_sqrt_len_tokens",
      "weightedmean_tokens",
      "last_token"
    ]

    for mode <- @pooling_modes do
      @mode mode

      test "matches Python for random model with #{@mode} pooling" do
        assert {:ok, %{dir: dir, python_output: python_output, text: text}} =
                 PythonBridge.generate_random_sentence_transformer(
                   pooling_mode: @mode,
                   with_normalize: false
                 )

        try do
          assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
          assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

          serving = SentenceTransformers.text_embedding(model_info, tokenizer)
          res = Nx.Serving.run(serving, text)

          assert Nx.shape(res.embedding) == Nx.shape(python_output)
          assert_all_close(res.embedding, python_output, atol: 1.0e-4)
        after
          File.rm_rf!(dir)
        end
      end
    end

    test "matches Python for random model with mean pooling and normalization" do
      assert {:ok, %{dir: dir, python_output: python_output, text: text}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 with_normalize: true
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python for random model with Dense projection layer" do
      assert {:ok, %{dir: dir, python_output: python_output, text: text}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 with_dense: true,
                 with_normalize: true
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python for random model with prompt and include_prompt=false" do
      prompt = "Represent the sentence for retrieval: "

      assert {:ok, %{dir: dir, python_output: python_output, text: text}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 prompt: prompt,
                 include_prompt: false,
                 text: "Hello world"
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python for random model with prompt and include_prompt=true" do
      prompt = "Represent the sentence for retrieval: "

      assert {:ok, %{dir: dir, python_output: python_output, text: text}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 prompt: prompt,
                 include_prompt: true,
                 text: "Hello world"
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python encode_queries using config_sentence_transformers.json prompt" do
      prompt = "Represent the query for retrieval: "
      text = "Hello world"

      assert {:ok, %{dir: dir, python_output: python_output}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 prompt: prompt,
                 prompt_name: "query",
                 text: text
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        [res] = SentenceTransformers.encode_queries(model_info, tokenizer, [text])

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python encode_corpus using config_sentence_transformers.json prompt" do
      query_prompt = "Represent the query for retrieval: "
      document_prompt = "Represent the document for retrieval: "
      text = "Hello world"

      assert {:ok, %{dir: dir, python_output: python_output}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 prompt: query_prompt,
                 document_prompt: document_prompt,
                 prompt_name: "document",
                 text: text
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        [res] = SentenceTransformers.encode_corpus(model_info, tokenizer, [text])

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end
  end

  describe "On-the-fly generated Dense / Normalize / LayerNorm / Dropout comparison" do
    test "matches Python for Dense with Tanh activation" do
      text = "Hello world"

      assert {:ok, %{dir: dir, python_output: python_output}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 with_dense: true,
                 dense_activation: "Tanh",
                 text: text
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    for activation <- ["Identity", "ReLU", "GELU", "SiLU", "Sigmoid", "Mish", "LeakyReLU"] do
      @activation activation

      test "matches Python for Dense with #{@activation} activation" do
        text = "Hello world"

        assert {:ok, %{dir: dir, python_output: python_output}} =
                 PythonBridge.generate_random_sentence_transformer(
                   pooling_mode: "mean_tokens",
                   with_dense: true,
                   dense_activation: @activation,
                   text: text
                 )

        try do
          assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
          assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

          serving = SentenceTransformers.text_embedding(model_info, tokenizer)
          res = Nx.Serving.run(serving, text)

          assert Nx.shape(res.embedding) == Nx.shape(python_output)
          assert_all_close(res.embedding, python_output, atol: 1.0e-4)
        after
          File.rm_rf!(dir)
        end
      end
    end

    test "matches Python for Normalize module" do
      text = "Hello world"

      assert {:ok, %{dir: dir, python_output: python_output}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 with_normalize: true,
                 text: text
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python for LayerNorm module" do
      text = "Hello world"

      assert {:ok, %{dir: dir, python_output: python_output}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 with_layernorm: true,
                 text: text
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python for Dropout module in eval mode" do
      text = "Hello world"

      assert {:ok, %{dir: dir, python_output: python_output}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 with_dropout: true,
                 text: text
               )

      try do
        assert {:ok, model_info} = SentenceTransformers.load_model({:local, dir})
        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    test "matches Python when fusing consecutive linear Dense layers" do
      text = "Hello world"

      assert {:ok, %{dir: dir, python_output: python_output}} =
               PythonBridge.generate_random_sentence_transformer(
                 pooling_mode: "mean_tokens",
                 with_fused_dense: true,
                 text: text
               )

      try do
        assert {:ok, model_info} =
                 SentenceTransformers.load_model({:local, dir}, fuse_dense: true)

        assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

        serving = SentenceTransformers.text_embedding(model_info, tokenizer)
        res = Nx.Serving.run(serving, text)

        assert Nx.shape(res.embedding) == Nx.shape(python_output)
        assert_all_close(res.embedding, python_output, atol: 1.0e-4)
      after
        File.rm_rf!(dir)
      end
    end

    for {description, opts} <- [
          {"first layer has bias", [fused_dense_first_bias: true]},
          {"first layer has activation", [fused_dense_first_activation: "Tanh"]},
          {"second layer has bias", [fused_dense_second_bias: true]},
          {"second layer has activation", [fused_dense_second_activation: "Tanh"]}
        ] do
      @description description
      @opts opts

      test "refuses to fuse when #{@description}" do
        text = "Hello world"

        assert {:ok, %{dir: dir, python_output: python_output}} =
                 PythonBridge.generate_random_sentence_transformer(
                   [pooling_mode: "mean_tokens", with_fused_dense: true, text: text] ++ @opts
                 )

        try do
          assert {:ok, model_info} =
                   SentenceTransformers.load_model({:local, dir}, fuse_dense: true)

          assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:local, dir})

          serving = SentenceTransformers.text_embedding(model_info, tokenizer)
          res = Nx.Serving.run(serving, text)

          assert Nx.shape(res.embedding) == Nx.shape(python_output)
          assert_all_close(res.embedding, python_output, atol: 1.0e-4)
        after
          File.rm_rf!(dir)
        end
      end
    end
  end

  describe "HuggingFace Hub SentenceTransformers comparison" do
    @tag :network
    test "matches Python for sentence-transformers/all-MiniLM-L6-v2" do
      repo_id = "sentence-transformers/all-MiniLM-L6-v2"
      input_text = "Hello world"

      # 1. Elixir SentenceTransformers execution
      assert {:ok, model_info} = SentenceTransformers.load_model({:hf, repo_id})
      assert {:ok, tokenizer} = SentenceTransformers.load_tokenizer({:hf, repo_id})

      serving = SentenceTransformers.text_embedding(model_info, tokenizer)
      elixir_res = Nx.Serving.run(serving, input_text)

      # 2. Python sentence_transformers execution
      python_res = PythonBridge.run_sentence_transformers(repo_id, input_text)

      # 3. Assertions and comparison
      assert Nx.shape(elixir_res.embedding) == Nx.shape(python_res)
      assert_all_close(elixir_res.embedding, python_res, atol: 1.0e-4)
    end
  end

  describe "HuggingFace tiny-random base models comparison" do
    @tiny_random_models [
      {"hf-internal-testing/tiny-random-BertModel", "last_hidden_state"},
      {"hf-internal-testing/tiny-random-RobertaModel", "last_hidden_state"}
    ]

    for {repo_id, output_key} <- @tiny_random_models do
      @repo_id repo_id
      @output_key output_key

      @tag :network
      test "matches Python transformers for #{@repo_id}" do
        inputs = %{
          "input_ids" => [[10, 20, 30, 40, 50]],
          "attention_mask" => [[1, 1, 1, 1, 1]]
        }

        elixir_inputs = %{
          "input_ids" => Nx.tensor(inputs["input_ids"]),
          "attention_mask" => Nx.tensor(inputs["attention_mask"])
        }

        assert {:ok, %{model: model, params: params}} = Bumblebee.load_model({:hf, @repo_id})
        elixir_outputs = Axon.predict(model, params, elixir_inputs)

        python_tensor =
          PythonBridge.run_hf_model(@repo_id, inputs, output_key: @output_key)

        elixir_tensor = Map.fetch!(elixir_outputs, :hidden_state)
        assert Nx.shape(elixir_tensor) == Nx.shape(python_tensor)
        assert_all_close(elixir_tensor, python_tensor, atol: 1.0e-4)
      end
    end
  end
end
