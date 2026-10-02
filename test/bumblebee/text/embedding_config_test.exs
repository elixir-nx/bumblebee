defmodule Bumblebee.Text.EmbeddingConfigTest do
  use ExUnit.Case, async: true

  alias Bumblebee.HuggingFace.Transformers.Config

  test "Gemma prefers its native bidirectional field" do
    spec = Bumblebee.configure(Bumblebee.Text.Gemma3Text)

    assert Config.load(spec, %{"is_causal" => true, "use_bidirectional_attention" => true}).use_bidirectional_attention
  end

  test "Qwen3 loads sliding layers and respects the upstream enable flag" do
    data = %{
      "num_hidden_layers" => 3,
      "use_sliding_window" => true,
      "sliding_window" => 2,
      "max_window_layers" => 1
    }

    spec = Config.load(Bumblebee.configure(Bumblebee.Text.Qwen3), data)
    assert spec.attention_window_size == 2
    assert spec.max_window_layers == 1
    disabled = Config.load(spec, Map.put(data, "use_sliding_window", false))
    assert disabled.attention_window_size == nil
    assert Config.load(spec, Map.delete(data, "use_sliding_window")).attention_window_size == nil
    assert Config.load(spec, Map.delete(data, "sliding_window")).attention_window_size == 4096

    assert_raise ArgumentError, ~r/use_sliding_window/, fn ->
      Config.load(spec, Map.put(data, "use_sliding_window", "false"))
    end

    explicit =
      Config.load(
        spec,
        Map.put(data, "layer_types", ["full_attention", "sliding_attention", "full_attention"])
      )

    assert explicit.layer_types == [:full_attention, :sliding_attention, :full_attention]
  end

  for module <- [
        Bumblebee.Text.Gemma3Text,
        Bumblebee.Text.Llama,
        Bumblebee.Text.Mistral,
        Bumblebee.Text.Qwen3
      ] do
    @module module

    test "#{inspect(module)} loads is_causal without changing the base default" do
      spec = Bumblebee.configure(@module, architecture: :base)
      refute Map.get(spec, :use_bidirectional_attention)
      assert Config.load(spec, %{"is_causal" => false}).use_bidirectional_attention
      refute Config.load(spec, %{"is_causal" => true}).use_bidirectional_attention

      assert Bumblebee.configure(spec, use_bidirectional_attention: true).use_bidirectional_attention
    end

    test "#{inspect(module)} rejects bidirectional generation cache" do
      spec = Config.load(Bumblebee.configure(@module), %{"is_causal" => false})

      assert_raise ArgumentError, ~r/bidirectional/, fn ->
        @module.init_cache(spec, 1, 8, %{})
      end
    end

    test "#{inspect(module)} rejects a directly supplied bidirectional cache" do
      spec =
        @module
        |> Bumblebee.configure(tiny_options(@module))
        |> Config.load(%{"is_causal" => false})

      model = Bumblebee.build_model(spec)

      cache =
        Bumblebee.Layers.Decoder.init_cache(1, 8,
          hidden_size: spec.hidden_size,
          attention_head_size: Map.get(spec, :attention_head_size),
          decoder_num_attention_heads: spec.num_attention_heads,
          decoder_num_key_value_heads: spec.num_key_value_heads,
          decoder_num_blocks: spec.num_blocks
        )

      input_template = %{"input_ids" => Nx.template({1, 1}, :s64)}
      {init_fun, predict_fun} = Axon.build(model, compiler: EXLA)
      params = init_fun.(input_template, Axon.ModelState.empty())

      assert_raise Axon.CompileError, ~r/bidirectional/, fn ->
        predict_fun.(params, %{
          "input_ids" => Nx.tensor([[1]]),
          "cache" => cache
        })
      end
    end
  end

  defp tiny_options(Bumblebee.Text.Gemma3Text) do
    [
      vocab_size: 32,
      max_positions: 8,
      hidden_size: 16,
      intermediate_size: 20,
      attention_head_size: 8,
      attention_scale_base: 8,
      num_blocks: 1,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      attention_window_size: 4,
      layer_types: [:full_attention]
    ]
  end

  defp tiny_options(Bumblebee.Text.Llama) do
    [
      vocab_size: 32,
      max_positions: 8,
      hidden_size: 16,
      intermediate_size: 20,
      attention_head_size: 8,
      num_blocks: 1,
      num_attention_heads: 2,
      num_key_value_heads: 1
    ]
  end

  defp tiny_options(Bumblebee.Text.Mistral) do
    [
      vocab_size: 32,
      max_positions: 8,
      hidden_size: 16,
      intermediate_size: 20,
      num_blocks: 1,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      attention_window_size: 4
    ]
  end

  defp tiny_options(Bumblebee.Text.Qwen3) do
    [
      vocab_size: 32,
      max_positions: 8,
      hidden_size: 16,
      intermediate_size: 20,
      attention_head_size: 8,
      num_blocks: 1,
      num_attention_heads: 2,
      num_key_value_heads: 1
    ]
  end
end
