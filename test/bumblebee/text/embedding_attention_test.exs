defmodule Bumblebee.Text.EmbeddingAttentionTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  alias Bumblebee.HuggingFace.Transformers.Config

  for module <- [
        Bumblebee.Text.Gemma3Text,
        Bumblebee.Text.Llama,
        Bumblebee.Text.Mistral,
        Bumblebee.Text.Qwen3
      ],
      bidirectional <- [false, true] do
    @module module
    @bidirectional bidirectional

    test "#{inspect(module)} attention boundaries and padding, bidirectional=#{bidirectional}" do
      spec = spec(@module, @bidirectional)
      inputs = %{"input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6, 7]])}

      {init, predict} =
        Axon.build(Bumblebee.build_model(spec),
          compiler: EXLA,
          global_layer_options: [output_attentions: true]
        )

      params = init.(inputs, Axon.ModelState.empty())
      output = predict.(params, inputs)

      radius = radius(@module, @bidirectional)
      weights = elem(output.attentions, 0) |> Nx.to_flat_list()

      expected =
        for _head <- 1..2, query <- 0..6, key <- 0..6 do
          allowed?(query, key, radius, @bidirectional)
        end

      assert Enum.map(weights, &(&1 > 0)) == expected

      if @module in [Bumblebee.Text.Gemma3Text, Bumblebee.Text.Qwen3] do
        global = elem(output.attentions, 1) |> Nx.to_flat_list()

        expected_global =
          for _head <- 1..2, query <- 0..6, key <- 0..6, do: @bidirectional or key <= query

        assert Enum.map(global, &(&1 > 0)) == expected_global
      end

      padded = Map.put(inputs, "attention_mask", Nx.tensor([[1, 1, 1, 1, 0, 0, 0]]))
      changed_padding = Map.put(padded, "input_ids", Nx.tensor([[1, 2, 3, 4, 10, 11, 12]]))

      assert_all_close(
        predict.(params, padded).hidden_state[[.., 0..3, ..]],
        predict.(params, changed_padding).hidden_state[[.., 0..3, ..]]
      )

      changed_future = Map.put(inputs, "input_ids", Nx.tensor([[1, 2, 3, 4, 5, 6, 8]]))
      before = output.hidden_state[[.., 5, ..]]
      after_change = predict.(params, changed_future).hidden_state[[.., 5, ..]]

      if @bidirectional do
        refute Nx.to_number(Nx.all_close(before, after_change, atol: 1.0e-6, rtol: 1.0e-6)) == 1
      else
        assert_all_close(before, after_change, atol: 1.0e-6, rtol: 1.0e-6)
      end
    end
  end

  for module <- [
        Bumblebee.Text.Gemma3Text,
        Bumblebee.Text.Llama,
        Bumblebee.Text.Mistral,
        Bumblebee.Text.Qwen3
      ] do
    @module module

    test "#{inspect(module)} causal cache matches full inference" do
      spec = spec(@module, false)
      model = Bumblebee.build_model(spec)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1]])
      }

      {init, predict} = Axon.build(model, compiler: EXLA)
      params = init.(inputs, Axon.ModelState.empty())
      full = predict.(params, inputs)
      cache = @module.init_cache(spec, 1, 7, %{})

      prefix =
        predict.(params, %{
          "input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6]]),
          "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1]]),
          "cache" => cache
        })

      last =
        predict.(params, %{
          "input_ids" => Nx.tensor([[7]]),
          "attention_mask" => Nx.tensor([[1]]),
          "position_ids" => Nx.tensor([[6]]),
          "cache" => prefix.cache
        })

      assert_all_close(last.hidden_state, full.hidden_state[[.., 6..6, ..]],
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end

    test "#{inspect(module)} cached generation matches uncached greedy decoding" do
      spec =
        Bumblebee.configure(spec(@module, false), architecture: :for_causal_language_modeling)

      model = Bumblebee.build_model(spec)
      inputs = %{"input_ids" => Nx.tensor([[1, 2]])}
      {init, predict} = Axon.build(model, compiler: EXLA)
      params = init.(inputs, Axon.ModelState.empty())

      {expected, _ids} =
        Enum.map_reduce(1..2, inputs["input_ids"], fn _, ids ->
          logits = predict.(params, %{"input_ids" => ids}).logits
          token = logits[[0, -1, ..]] |> Nx.argmax() |> Nx.to_number()
          {token, Nx.concatenate([ids, Nx.tensor([[token]])], axis: 1)}
        end)

      config =
        Bumblebee.configure(Bumblebee.Text.GenerationConfig, max_new_tokens: 2, pad_token_id: 0)

      generate = Bumblebee.Text.Generation.build_generate(model, spec, config)
      output = generate.(params, Map.put(inputs, "seed", Nx.tensor([0])))
      assert_equal(output.token_ids, Nx.tensor([expected]))
    end
  end

  defp allowed?(_query, _key, nil, true), do: true
  defp allowed?(query, key, nil, false), do: key <= query
  defp allowed?(query, key, radius, true), do: abs(query - key) <= radius
  defp allowed?(query, key, radius, false), do: key <= query and query - key <= radius

  defp radius(Bumblebee.Text.Llama, _), do: nil
  defp radius(Bumblebee.Text.Gemma3Text, true), do: 2
  defp radius(_, true), do: 4
  defp radius(_, false), do: 3

  defp spec(module, bidirectional) do
    data = %{
      "vocab_size" => 32,
      "hidden_size" => 16,
      "intermediate_size" => 20,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 1,
      "head_dim" => 8,
      "query_pre_attn_scalar" => 8,
      "max_position_embeddings" => 16,
      "sliding_window" => 4,
      "use_sliding_window" => true,
      "max_window_layers" => 0,
      "is_causal" => not bidirectional,
      "use_bidirectional_attention" => bidirectional
    }

    data =
      if module in [Bumblebee.Text.Gemma3Text, Bumblebee.Text.Qwen3],
        do: Map.put(data, "layer_types", ["sliding_attention", "full_attention"]),
        else: data

    Config.load(Bumblebee.configure(module), data)
  end
end
