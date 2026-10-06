defmodule Bumblebee.SentenceTransformers.DenseEdgeTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.SentenceTransformers
  alias Bumblebee.SentenceTransformers.Downloader
  alias Bumblebee.SentenceTransformers.Pipeline

  test "last-token pooling handles left padding, prompt exclusion, and empty masks" do
    hidden = Axon.input("hidden", shape: {nil, nil, 2})
    mask = Axon.input("attention_mask", shape: {nil, nil})

    assert {:ok, model} =
             Pipeline.pooling_layer(
               hidden,
               mask,
               %{"pooling_mode_lasttoken" => true, "include_prompt" => false},
               "last"
             )

    inputs = %{
      "hidden" =>
        Nx.tensor([
          [[99.0, 99.0], [1.0, 2.0], [3.0, 4.0]],
          [[5.0, 6.0], [7.0, 8.0], [99.0, 99.0]]
        ]),
      "attention_mask" => Nx.tensor([[0, 1, 1], [1, 1, 0]]),
      "prompt_length" => Nx.tensor([1, 1])
    }

    {init, predict} = Axon.build(model)
    params = init.(inputs, Axon.ModelState.empty())
    assert_all_close(predict.(params, inputs), Nx.tensor([[3.0, 4.0], [7.0, 8.0]]))
  end

  @tag :tmp_dir
  test "root prompts and arbitrary first-module token limits are merged", %{tmp_dir: dir} do
    File.mkdir_p!(Path.join(dir, "encoder"))

    File.write!(
      Path.join(dir, "modules.json"),
      Jason.encode!([%{"idx" => 0, "path" => "encoder", "type" => "Transformer"}])
    )

    File.write!(
      Path.join(dir, "config_sentence_transformers.json"),
      Jason.encode!(%{"prompts" => %{"task" => "hello "}, "default_prompt_name" => "task"})
    )

    File.write!(
      Path.join(dir, "encoder/sentence_bert_config.json"),
      Jason.encode!(%{"max_seq_length" => 3})
    )

    assert {:ok, config} = SentenceTransformers.load_config({:local, dir})
    assert config.max_seq_length == 3
    assert config.default_prompt_name == "task"
    assert config.prompts == %{"task" => "hello "}

    File.rm!(Path.join(dir, "encoder/sentence_bert_config.json"))

    File.write!(
      Path.join(dir, "encoder/tokenizer_config.json"),
      Jason.encode!(%{"model_max_length" => 5})
    )

    assert {:ok, config} = SentenceTransformers.load_config({:local, dir})
    assert config.max_seq_length == 5
  end

  @tag :tmp_dir
  test "saved truncation stays separate from compilation padding and explicit overrides", %{
    tmp_dir: dir
  } do
    tokenizer = tiny_tokenizer(dir)
    ids = Axon.input("input_ids", shape: {nil, nil})
    mask = Axon.input("attention_mask", shape: {nil, nil})

    model =
      Axon.layer(
        fn ids, mask, _opts ->
          ids |> Nx.multiply(mask) |> Nx.sum(axes: [1], keep_axes: true) |> Nx.as_type(:f32)
        end,
        [ids, mask]
      )

    info = %{
      model: model,
      params: Axon.ModelState.empty(),
      spec: nil,
      sentence_transformers: %{max_seq_length: 3}
    }

    text = "hello world hello world hello world"

    for opts <- [
          [],
          [compile: [batch_size: 1, sequence_length: 8]],
          [compile: [batch_size: 1, sequence_length: [4, 8]]]
        ] do
      assert %{embedding: embedding} =
               Nx.Serving.run(SentenceTransformers.text_embedding(info, tokenizer, opts), text)

      assert Nx.to_flat_list(embedding) == [4.0]

      assert %{embedding: embedding} =
               Nx.Serving.run(
                 SentenceTransformers.text_embedding(
                   info,
                   tokenizer,
                   Keyword.put(opts, :prompt, "world ")
                 ),
                 text
               )

      assert Nx.to_flat_list(embedding) == [5.0]
    end

    assert %{embedding: embedding} =
             Nx.Serving.run(
               SentenceTransformers.text_embedding(info, tokenizer, max_seq_length: 5),
               text
             )

    assert Nx.to_flat_list(embedding) == [7.0]

    configured =
      SentenceTransformers.configure_embedding_tokenizer(info, tokenizer,
        compile: [batch_size: 1, sequence_length: 8]
      )

    tokens = SentenceTransformers.apply_embedding_tokenizer(configured, [text], 8)
    assert Nx.to_list(tokens["attention_mask"]) == [[1, 1, 1, 0, 0, 0, 0, 0]]
  end

  @tag :tmp_dir
  test "saved limits truncate without forcing padding or bypassing smaller buckets", %{
    tmp_dir: dir
  } do
    tokenizer = tiny_tokenizer(dir)
    info = %{sentence_transformers: %{max_seq_length: 512}}

    dynamic = SentenceTransformers.configure_embedding_tokenizer(info, tokenizer, [])

    assert Nx.shape(
             SentenceTransformers.apply_embedding_tokenizer(dynamic, ["hello"], nil)["input_ids"]
           ) == {1, 1}

    compiled =
      SentenceTransformers.configure_embedding_tokenizer(info, tokenizer,
        compile: [batch_size: 1, sequence_length: [8, 512]]
      )

    inputs = SentenceTransformers.apply_embedding_tokenizer(compiled, ["hello"], [8, 512])
    assert Nx.shape(inputs["input_ids"]) == {1, 8}
    assert Nx.to_list(inputs["attention_mask"]) == [[1, 0, 0, 0, 0, 0, 0, 0]]

    long_text = Enum.join(List.duplicate("hello", 600), " ")

    assert Nx.shape(
             SentenceTransformers.apply_embedding_tokenizer(dynamic, [long_text], nil)[
               "input_ids"
             ]
           ) ==
             {1, 512}
  end

  @tag :tmp_dir
  test "compiled padding supports EOS-only tokenizers", %{tmp_dir: dir} do
    tokenizer = %{tiny_tokenizer(dir) | special_tokens: %{eos: "[PAD]", unk: "[UNK]"}}
    info = %{sentence_transformers: %{max_seq_length: 512}}

    for direction <- [:left, :right] do
      tokenizer = %{tokenizer | pad_direction: direction}

      configured =
        SentenceTransformers.configure_embedding_tokenizer(info, tokenizer,
          compile: [batch_size: 1, sequence_length: [8, 512]]
        )

      inputs = SentenceTransformers.apply_embedding_tokenizer(configured, ["hello"], [8, 512])

      expected =
        if direction == :left, do: [0, 0, 0, 0, 0, 0, 0, 1], else: [1, 0, 0, 0, 0, 0, 0, 0]

      assert Nx.to_list(inputs["input_ids"]) == [expected]
      assert Nx.to_list(inputs["attention_mask"]) == [expected]
    end
  end

  @tag :tmp_dir
  test "malformed manifests and download failures propagate from model and tokenizer loading", %{
    tmp_dir: dir
  } do
    File.write!(Path.join(dir, "modules.json"), "invalid json")
    assert {:error, _} = SentenceTransformers.load_model({:local, dir})
    assert {:error, _} = SentenceTransformers.load_tokenizer({:local, dir})

    assert {:error, "invalid repository format: :invalid"} =
             SentenceTransformers.load_model(:invalid)

    assert {:error, "invalid repository format: :invalid"} =
             SentenceTransformers.load_tokenizer(:invalid)

    refute Downloader.missing_file?(
             {:hf, "repo"},
             "modules.json",
             "HTTP request failed with status 500, url: test"
           )

    refute Downloader.missing_file?(
             {:hf, "repo"},
             "modules.json",
             "repository not found, url: test"
           )

    refute Downloader.missing_file?({:hf, "repo"}, "modules.json", "cache file not found")

    assert Downloader.missing_file?(
             {:hf, "repo"},
             "modules.json",
             "file not found, url: test"
           )

    assert Downloader.missing_file?({:local, dir}, "absent.json", "missing")
  end

  defp tiny_tokenizer(dir) do
    path = Path.join(dir, "tokenizer.json")

    File.write!(
      path,
      Jason.encode!(%{
        "version" => "1.0",
        "truncation" => nil,
        "padding" => nil,
        "added_tokens" => [],
        "normalizer" => nil,
        "pre_tokenizer" => %{"type" => "Whitespace"},
        "post_processor" => nil,
        "decoder" => nil,
        "model" => %{
          "type" => "WordLevel",
          "vocab" => %{"[PAD]" => 0, "hello" => 1, "world" => 2, "[UNK]" => 3},
          "unk_token" => "[UNK]"
        }
      })
    )

    {:ok, native} = Tokenizers.Tokenizer.from_file(path)

    %Bumblebee.Text.PreTrainedTokenizer{
      native_tokenizer: native,
      type: :bert,
      special_tokens: %{pad: "[PAD]", unk: "[UNK]"},
      add_special_tokens: false
    }
  end
end
