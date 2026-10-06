defmodule Bumblebee.SentenceTransformers.PythonEnvironmentTest do
  use ExUnit.Case, async: false

  alias Bumblebee.SentenceTransformers.TestSupport.PythonBridge

  @tag :python
  @tag :tmp_dir
  test "incompatible Python is checked once and reports an actionable version error", %{
    tmp_dir: dir
  } do
    python = System.find_executable("python3")
    assert python, "python3 is needed for the Python environment validation regression"
    wrapper = Path.join(dir, "incompatible-python")
    count_file = Path.join(dir, "invocations")

    File.write!(wrapper, """
    #!#{python}
    import importlib.metadata, sys
    with open(#{Jason.encode!(count_file)}, "a") as counter:
        counter.write("checked\\n")
    importlib.metadata.version = lambda package: "0.0.0"
    exec(sys.argv[-1])
    """)

    File.chmod!(wrapper, 0o755)
    previous = System.get_env("SENTENCE_TRANSFORMERS_PYTHON") || System.get_env("STING_PYTHON")
    System.put_env("SENTENCE_TRANSFORMERS_PYTHON", wrapper)

    on_exit(fn ->
      if previous,
        do: System.put_env("SENTENCE_TRANSFORMERS_PYTHON", previous),
        else: System.delete_env("SENTENCE_TRANSFORMERS_PYTHON")
    end)

    for _ <- 1..2 do
      assert_raise RuntimeError,
                   ~r/expected sentence-transformers==6.1.0, found 0.0.0.*(SENTENCE_TRANSFORMERS_PYTHON|STING_PYTHON)/,
                   fn ->
                     PythonBridge.run_python_code(
                       "raise RuntimeError('must not execute')",
                       ["sentence-transformers"]
                     )
                   end
    end

    assert File.read!(count_file) == "checked\n"
  end
end
