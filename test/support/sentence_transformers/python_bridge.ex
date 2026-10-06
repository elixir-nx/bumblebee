defmodule Bumblebee.SentenceTransformers.TestSupport.PythonBridge do
  @moduledoc """
  Helper module for executing Python inference scripts (via `uv` or `python3`)
  and generating random SentenceTransformers / Hugging Face model checkpoints
  to compare outputs with Bumblebee.SentenceTransformers.
  """

  @doc """
  Runs inference using Python's `sentence_transformers` library on a given model and text,
  returning the computed embeddings as an `Nx.Tensor`.
  """
  def run_sentence_transformers(repo_id, text, opts \\ []) do
    code = """
    import json
    import os
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("#{repo_id}")
    encode_opts = json.loads(#{inspect(Jason.encode!(Map.new(Keyword.take(opts, [:prompt, :prompt_name]))))})
    res = model.encode(#{inspect(text)}, **encode_opts)
    print("__RESULT__:" + json.dumps(res.tolist()))
    """

    deps = ["sentence-transformers", "torch"]
    run_python(code, deps, opts)
  end

  @pooling_mode_flags %{
    "mean_tokens" => "pooling_mode_mean_tokens=True",
    "cls_token" => "pooling_mode_cls_token=True",
    "max_tokens" => "pooling_mode_max_tokens=True",
    "mean_sqrt_len_tokens" => "pooling_mode_mean_sqrt_len_tokens=True",
    "weightedmean_tokens" => "pooling_mode_weightedmean_tokens=True",
    "last_token" => "pooling_mode_lasttoken=True"
  }

  @doc """
  Generates an on-the-fly random SentenceTransformer model checkpoint in Python with random weights,
  saves it to a temporary directory, executes inference on `text`, and returns:
  `{:ok, %{dir: dir, python_output: nx_tensor, text: text}}`.

  ## Options
    * `:pooling_mode` - a single pooling mode name or a list of mode names.
      Supported: `"mean_tokens"`, `"cls_token"`, `"max_tokens"`,
      `"mean_sqrt_len_tokens"`, `"weightedmean_tokens"`, `"last_token"`
      (default: `["mean_tokens"]`).
    * `:with_dense` - whether to include a Dense projection layer (default: `false`)
    * `:dense_activation` - activation for the Dense layer: `"Identity"`, `"Tanh"`,
      `"ReLU"`, `"GELU"`, `"SiLU"`, `"Sigmoid"`, `"Mish"`, `"LeakyReLU"`
      (default: `"Tanh"`)
    * `:with_normalize` - whether to append a Normalize module (default: `false`)
    * `:with_layernorm` - whether to append a LayerNorm module (default: `false`)
    * `:with_dropout` - whether to append a Dropout module (default: `false`)
    * `:with_fused_dense` - whether to append two consecutive Dense layers
      (default: `false`). By default both use `Identity` activation and no bias so
      Bumblebee.SentenceTransformers can fuse them.
    * `:fused_dense_first_bias` - whether the first fused Dense layer has bias
      (default: `false`)
    * `:fused_dense_first_activation` - activation for the first fused Dense layer
      (default: `"Identity"`)
    * `:fused_dense_second_bias` - whether the second fused Dense layer has bias
      (default: `false`)
    * `:fused_dense_second_activation` - activation for the second fused Dense layer
      (default: `"Identity"`)
    * `:prompt` - prompt string to prepend to the input (default: `nil`)
    * `:prompt_name` - name of a prompt in `config_sentence_transformers.json` to
      use for the Python encode call (default: `nil`)
    * `:document_prompt` - prompt string to register under the `"document"` name
      (default: `nil`)
    * `:include_prompt` - whether to include prompt tokens in pooling
      (default: `true`). Only relevant when `:prompt` is given.
    * `:text` - text to encode (default: `"Hello world"`)
  """
  def generate_random_sentence_transformer(opts \\ []) do
    pooling_mode = Keyword.get(opts, :pooling_mode, ["mean_tokens"])
    with_dense = Keyword.get(opts, :with_dense, false)
    dense_activation = Keyword.get(opts, :dense_activation, "Tanh")
    with_normalize = Keyword.get(opts, :with_normalize, false)
    with_layernorm = Keyword.get(opts, :with_layernorm, false)
    with_dropout = Keyword.get(opts, :with_dropout, false)
    with_fused_dense = Keyword.get(opts, :with_fused_dense, false)
    fused_dense_first_bias = Keyword.get(opts, :fused_dense_first_bias, false)
    fused_dense_first_activation = Keyword.get(opts, :fused_dense_first_activation, "Identity")
    fused_dense_second_bias = Keyword.get(opts, :fused_dense_second_bias, false)
    fused_dense_second_activation = Keyword.get(opts, :fused_dense_second_activation, "Identity")
    prompt = Keyword.get(opts, :prompt)
    include_prompt = Keyword.get(opts, :include_prompt, true)
    text = Keyword.get(opts, :text, "Hello world")
    max_seq_length = Keyword.get(opts, :max_seq_length, 64)

    mode_flags =
      pooling_mode
      |> List.wrap()
      |> Enum.map_join(", ", &Map.fetch!(@pooling_mode_flags, &1))

    mode_flags =
      if "mean_tokens" in List.wrap(pooling_mode),
        do: mode_flags,
        else: "pooling_mode_mean_tokens=False, " <> mode_flags

    pooling_args =
      if prompt do
        "#{mode_flags}, include_prompt=#{if include_prompt, do: "True", else: "False"}"
      else
        mode_flags
      end

    prompt_name = Keyword.get(opts, :prompt_name)

    encode_call =
      cond do
        prompt_name ->
          "st_model.encode(#{inspect(text)}, prompt_name=#{inspect(prompt_name)})"

        prompt ->
          "st_model.encode(#{inspect(text)}, prompt=#{inspect(prompt)})"

        true ->
          "st_model.encode(#{inspect(text)})"
      end

    prompts =
      if prompt do
        %{"query" => prompt}
      else
        %{}
      end

    prompts =
      if document_prompt = Keyword.get(opts, :document_prompt) do
        Map.put(prompts, "document", document_prompt)
      else
        prompts
      end

    st_config =
      if map_size(prompts) > 0 do
        Jason.encode!(%{
          "prompts" => prompts,
          "default_prompt_name" => "query"
        })
      else
        Jason.encode!(%{})
      end

    code = """
    import json
    import os
    import sys
    import tempfile
    import warnings
    warnings.filterwarnings("ignore")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from sentence_transformers import SentenceTransformer, models

    torch.manual_seed(42)

    hidden_size = 32
    config = BertConfig(
        vocab_size=128,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64
    )
    base_model = BertModel(config)

    with tempfile.TemporaryDirectory() as tmp_base:
        base_model.save_pretrained(tmp_base)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world",
                 "represent", "the", "sentence", "for", "retrieval", ":", "query", "document"]
        vocab += [f"unused_{i}" for i in range(128 - len(vocab))]
        with open(os.path.join(tmp_base, "vocab.txt"), "w") as f:
            f.write("\\n".join(vocab))
        BertTokenizerFast(vocab={token: i for i, token in enumerate(vocab)}, do_lower_case=True).save_pretrained(tmp_base)

        word_embedding_model = models.Transformer(tmp_base, max_seq_length=#{max_seq_length})
        pooling_model = models.Pooling(
            word_embedding_model.get_embedding_dimension(),
            #{pooling_args}
        )

        modules = [word_embedding_model, pooling_model]

        activation_map = {
            "Identity": torch.nn.Identity(),
            "Tanh": torch.nn.Tanh(),
            "ReLU": torch.nn.ReLU(),
            "GELU": torch.nn.GELU(),
            "SiLU": torch.nn.SiLU(),
            "Sigmoid": torch.nn.Sigmoid(),
            "Mish": torch.nn.Mish(),
            "LeakyReLU": torch.nn.LeakyReLU()
        }

        if #{if with_dense, do: "True", else: "False"}:
            dense_model = models.Dense(
                in_features=hidden_size,
                out_features=16,
                bias=True,
                activation_function=activation_map[#{inspect(dense_activation)}]
            )
            modules.append(dense_model)

        if #{if with_fused_dense, do: "True", else: "False"}:
            modules.append(models.Dense(
                in_features=hidden_size,
                out_features=24,
                bias=#{if fused_dense_first_bias, do: "True", else: "False"},
                activation_function=activation_map[#{inspect(fused_dense_first_activation)}]
            ))
            modules.append(models.Dense(
                in_features=24,
                out_features=16,
                bias=#{if fused_dense_second_bias, do: "True", else: "False"},
                activation_function=activation_map[#{inspect(fused_dense_second_activation)}]
            ))

        if #{if with_normalize, do: "True", else: "False"}:
            modules.append(models.Normalize())

        if #{if with_layernorm, do: "True", else: "False"}:
            modules.append(models.LayerNorm(hidden_size))

        if #{if with_dropout, do: "True", else: "False"}:
            modules.append(models.Dropout(0.5))

        st_model = SentenceTransformer(modules=modules)

        save_dir = tempfile.mkdtemp(prefix="st_random_st_")
        st_model.save(save_dir)

        with open(save_dir + "/config_sentence_transformers.json", "w") as f:
            f.write(#{inspect(st_config)})

        st_model = SentenceTransformer(save_dir)

        emb = #{encode_call}.tolist()

        print("__DIR__:" + save_dir)
        print("__RESULT__:" + json.dumps(emb))
    """

    deps = ["sentence-transformers", "torch"]
    {cmd, args} = python_exec_args(code, deps)

    case System.cmd(cmd, args, stderr_to_stdout: true) do
      {output, 0} ->
        [_, dir] = Regex.run(~r/__DIR__:(.*)$/m, output)
        [_, result_json] = Regex.run(~r/__RESULT__:(.*)$/m, output)

        python_output = Nx.tensor(Jason.decode!(result_json), type: :f32)
        {:ok, %{dir: String.trim(dir), python_output: python_output, text: text}}

      {output, exit_code} ->
        raise "Failed to generate random SentenceTransformer (exit #{exit_code}):\n#{output}"
    end
  end

  @doc """
  Runs inference using Python's `transformers` library on a given model and inputs.
  """
  def run_hf_model(repo_id, inputs, opts \\ []) do
    output_key = Keyword.get(opts, :output_key, "last_hidden_state")
    json_inputs = Jason.encode!(inputs)

    code = """
    import json
    import os
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    import torch
    from transformers import AutoModel

    raw_inputs = json.loads(#{inspect(json_inputs)})
    inputs = {k: torch.tensor(v) for k, v in raw_inputs.items()}

    model = AutoModel.from_pretrained("#{repo_id}")
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        result = getattr(outputs, "#{output_key}", None)
        if result is None and "#{output_key}" in outputs:
            result = outputs["#{output_key}"]
        if hasattr(result, "cpu"):
            result = result.cpu().numpy().tolist()

    print("__RESULT__:" + json.dumps(result))
    """

    deps = ["transformers", "torch"]
    run_python(code, deps, opts)
  end

  @doc """
  Generates a tiny model with completely random weights on the fly in Python, saves it to
  a temporary directory, executes inference in Python, and returns:
  `%{dir: temp_dir, inputs: inputs_map, python_output: nx_tensor}`.
  """
  def generate_random_hf_model(model_family, opts \\ []) do
    seq_len = Keyword.get(opts, :seq_len, 5)
    batch_size = Keyword.get(opts, :batch_size, 1)

    code = """
    import json
    import os
    import sys
    import tempfile
    import warnings
    warnings.filterwarnings("ignore")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    import torch
    from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel, GPT2Config, GPT2Model

    family = "#{model_family}"
    batch_size = #{batch_size}
    seq_len = #{seq_len}

    torch.manual_seed(42)

    if family == "bert":
        config = BertConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64
        )
        model = BertModel(config)
        inputs = {
            "input_ids": torch.randint(0, 128, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long)
        }
        output_attr = "last_hidden_state"

    elif family == "roberta":
        config = RobertaConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
            type_vocab_size=1
        )
        model = RobertaModel(config)
        inputs = {
            "input_ids": torch.randint(0, 128, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long)
        }
        output_attr = "last_hidden_state"

    elif family == "gpt2":
        config = GPT2Config(
            vocab_size=128,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_positions=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2
        )
        model = GPT2Model(config)
        inputs = {
            "input_ids": torch.randint(0, 128, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long)
        }
        output_attr = "last_hidden_state"
    else:
        raise ValueError(f"Unsupported family: {family}")

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        result = getattr(outputs, output_attr).cpu().numpy().tolist()

    temp_dir = tempfile.mkdtemp(prefix="st_random_hf_")
    model.save_pretrained(temp_dir)

    raw_inputs = {k: v.tolist() for k, v in inputs.items()}

    print("__DIR__:" + temp_dir)
    print("__INPUTS__:" + json.dumps(raw_inputs))
    print("__RESULT__:" + json.dumps(result))
    """

    deps = ["transformers", "torch"]
    {cmd, args} = python_exec_args(code, deps)

    case System.cmd(cmd, args, stderr_to_stdout: true) do
      {output, 0} ->
        [_, dir] = Regex.run(~r/__DIR__:(.*)$/m, output)
        [_, result_json] = Regex.run(~r/__RESULT__:(.*)$/m, output)

        inputs =
          Regex.run(~r/__INPUTS__:(.*)$/m, output)
          |> Enum.at(1)
          |> Jason.decode!()
          |> Map.new(fn {k, v} -> {k, Nx.tensor(v)} end)

        python_output = Nx.tensor(Jason.decode!(result_json), type: :f32)

        {:ok, %{dir: String.trim(dir), inputs: inputs, python_output: python_output}}

      {output, exit_code} ->
        raise "Failed to generate random model (exit #{exit_code}):\n#{output}"
    end
  end

  @doc """
  Runs custom Python code that prints `__RESULT__:<json>` and returns the parsed Nx tensor or raw data.
  """
  def run_python_code(code, deps \\ ["sentence-transformers", "torch"], opts \\ []) do
    {cmd, args} = python_exec_args(code, deps)

    case System.cmd(cmd, args, stderr_to_stdout: true) do
      {output, 0} ->
        parse_result(output, opts)

      {output, exit_code} ->
        raise "Python process failed with exit code #{exit_code}:\n#{output}"
    end
  end

  defp run_python(code, deps, opts) do
    run_python_code(code, deps, opts)
  end

  defp python_exec_args(code, deps) do
    {cmd, args} = python_command(deps)
    verify_environment!(cmd, args, deps)
    {cmd, args ++ ["-c", code]}
  end

  defp verify_environment!(cmd, args, deps) do
    key = {__MODULE__, :environment, cmd, args, deps}

    result =
      case :persistent_term.get(key, :unchecked) do
        :unchecked ->
          code = """
          import importlib.metadata as metadata
          import json, platform, sys
          required = #{Jason.encode!(deps)}
          versions = {}
          try:
              for package in required:
                  versions[package] = metadata.version(package)
              if "sentence-transformers" in versions and versions["sentence-transformers"] != "6.1.0":
                  raise RuntimeError("expected sentence-transformers==6.1.0, found " + versions["sentence-transformers"])
              if "torch" in required:
                  import torch
                  versions["torch_runtime"] = torch.__version__
              print(json.dumps({"python": platform.python_version(), "platform": platform.platform(), "dependencies": versions}))
          except Exception as error:
              sys.exit("SentenceTransformers parity setup: " + str(error) + ". Install sentence-transformers==6.1.0, torch, numpy and datasets in a venv and set SENTENCE_TRANSFORMERS_PYTHON (or STING_PYTHON), or install uv.")
          """

          result = System.cmd(cmd, args ++ ["-c", code], stderr_to_stdout: true)
          :persistent_term.put(key, result)

          if elem(result, 1) == 0 do
            versions =
              for app <- [:bumblebee, :axon, :nx],
                  into: %{},
                  do: {app, to_string(Application.spec(app, :vsn))}

            IO.puts(
              "SentenceTransformers parity runtime: " <>
                Jason.encode!(%{
                  elixir: System.version(),
                  otp: System.otp_release(),
                  backend: inspect(Nx.default_backend()),
                  dependencies: versions
                })
            )

            IO.puts("SentenceTransformers Python runtime: " <> String.trim(elem(result, 0)))
          end

          result

        cached ->
          cached
      end

    case result do
      {_, 0} -> :ok
      {output, _} -> raise String.trim(output)
    end
  end

  defp python_command(deps) do
    cond do
      python_path =
          System.get_env("SENTENCE_TRANSFORMERS_PYTHON") || System.get_env("STING_PYTHON") ->
        {python_path, []}

      uv_path = System.find_executable("uv") ->
        with_args =
          Enum.flat_map(deps, fn dep ->
            dep = if dep == "sentence-transformers", do: "sentence-transformers==6.1.0", else: dep
            ["--with", dep]
          end)

        {uv_path, ["run", "--quiet"] ++ with_args ++ ["python"]}

      py3_path = System.find_executable("python3") ->
        {py3_path, []}

      true ->
        raise "Neither 'uv' nor 'python3' executable was found in PATH"
    end
  end

  defp parse_result(output, opts) do
    case Regex.run(~r/__RESULT__:(.*)$/m, output) do
      [_, json_data] ->
        data = Jason.decode!(json_data)
        type = Keyword.get(opts, :type, :f32)

        if type == :raw do
          data
        else
          Nx.tensor(data, type: type)
        end

      nil ->
        raise "Could not find __RESULT__ marker in Python output:\n#{output}"
    end
  end
end
