defmodule BumblebeeTest do
  use ExUnit.Case, async: true

  describe "load_model/2" do
    @tag :capture_log
    test "supports sharded params" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"})

      # PyTorch format

      assert {:ok, %{params: sharded_params}} =
               Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-GPT2Model-sharded"})

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(sharded_params))

      # Safetensors

      assert {:ok, %{params: sharded_params}} =
               Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-GPT2Model-sharded"},
                 params_filename: "model.safetensors"
               )

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(sharded_params))
    end

    test "supports .safetensors params" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"})

      assert {:ok, %{params: safetensors_params}} =
               Bumblebee.load_model(
                 {:hf, "bumblebee-testing/tiny-random-GPT2Model-safetensors-only"}
               )

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(safetensors_params))
    end

    test "supports params variants" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-bert-variant"},
                 params_variant: "v2"
               )

      assert {:ok, %{params: sharded_params}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-bert-variant-sharded"},
                 params_variant: "v2"
               )

      assert Enum.sort(Map.keys(params)) == Enum.sort(Map.keys(sharded_params))

      assert_raise ArgumentError,
                   ~s/parameters variant "v3" not found, available variants: "v2"/,
                   fn ->
                     Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-bert-variant"},
                       params_variant: "v3"
                     )
                   end

      assert_raise ArgumentError,
                   ~s/parameters variant "v3" not found, available variants: "v2"/,
                   fn ->
                     Bumblebee.load_model(
                       {:hf, "hf-internal-testing/tiny-random-bert-variant-sharded"},
                       params_variant: "v3"
                     )
                   end
    end

    test "passing :type casts params accordingly" do
      assert {:ok, %{params: %Axon.ModelState{data: params}}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"},
                 type: :bf16
               )

      assert Nx.type(params["decoder.blocks.0.ffn.output"]["kernel"]) == {:bf, 16}
      assert Nx.type(params["decoder.blocks.0.ffn.output"]["bias"]) == {:bf, 16}

      assert {:ok, %{params: %Axon.ModelState{data: params}}} =
               Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2Model"},
                 type: Axon.MixedPrecision.create_policy(params: :f16)
               )

      assert Nx.type(params["decoder.blocks.0.ffn.output"]["kernel"]) == {:f, 16}
      assert Nx.type(params["decoder.blocks.0.ffn.output"]["bias"]) == {:f, 16}
    end

    test "uses :safetensors_reader to read .safetensors files" do
      test_pid = self()

      reader = fn path ->
        send(test_pid, {:reader_called, path})
        Safetensors.read!(path, lazy: true)
      end

      assert {:ok, %{params: params}} =
               Bumblebee.load_model(
                 {:hf, "bumblebee-testing/tiny-random-GPT2Model-safetensors-only"},
                 safetensors_reader: reader
               )

      assert_received {:reader_called, path}
      assert File.exists?(path)

      assert {:ok, %{params: default_params}} =
               Bumblebee.load_model(
                 {:hf, "bumblebee-testing/tiny-random-GPT2Model-safetensors-only"}
               )

      assert Enum.sort(Map.keys(params.data)) == Enum.sort(Map.keys(default_params.data))
    end

    @tag :tmp_dir
    test "loads spec from local directory containing subdirectories", %{tmp_dir: tmp_dir} do
      File.mkdir_p!(Path.join(tmp_dir, "sub/dir"))
      File.write!(Path.join(tmp_dir, "config.json"), ~s/{"architectures": ["BertModel"]}/)
      File.write!(Path.join(tmp_dir, "sub/dir/extra.json"), ~s/{}/)

      assert {:ok, %Bumblebee.Text.Bert{}} = Bumblebee.load_spec({:local, tmp_dir})
    end

    @tag :tmp_dir
    test "loads spec from local directory with :subdir option", %{tmp_dir: tmp_dir} do
      sub = Path.join(tmp_dir, "sub_model")
      File.mkdir_p!(sub)
      File.write!(Path.join(sub, "config.json"), ~s/{"architectures": ["BertModel"]}/)

      assert {:ok, %Bumblebee.Text.Bert{}} =
               Bumblebee.load_spec({:local, tmp_dir, subdir: "sub_model"})
    end

    @tag :tmp_dir
    test "limits directory traversal to :max_depth", %{tmp_dir: tmp_dir} do
      deep_dir = Path.join(tmp_dir, "l1/l2/l3")
      File.mkdir_p!(deep_dir)
      File.write!(Path.join(deep_dir, "config.json"), ~s/{"architectures": ["BertModel"]}/)

      assert_raise ArgumentError, ~r/no config file found in the given repository/, fn ->
        Bumblebee.load_spec({:local, tmp_dir, max_depth: 2})
      end

      assert {:ok, %Bumblebee.Text.Bert{}} =
               Bumblebee.load_spec({:local, tmp_dir, max_depth: 3, subdir: "l1/l2/l3"})
    end
  end
end
